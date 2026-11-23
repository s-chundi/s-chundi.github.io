---
layout: post
author: Suhas Chundi
---

<details>
  <summary>Author's Notes</summary>
  
  More freeform notes from my project to implement a mini distributed training platform. I've completed some of the implementation but don't think I'll get around to a completed post.
</details>

### Ray and Asyncio Analogies

|Term|Definition|Analogy|
|---|---|---|
| Ray | Distributed runtime across processes/machines | Corporate |
| Asyncio | Event loop w/in a single thread | A satellite office |
| Worker | A Python process | Employee or Supervisor |
| Task | `Ray.remote()` on a fn | Job to do |
| Actor | Stateful object in process | (state) Notebook and set of jobs to do |
| Actor method | Work for an actor instance | A job w/ a notebook update |
| ObjectRef | Future for a Ray task's result | A ticket for work in another office |
| `asyncio.Queue` | Bounded async queue | A fixed-size loading dock w/in an office |
| `asyncio.Condition` | Lock + wait/notify on shared state | A bell rung when the notice board changes |

## Asyncio and Ray

We have a queue with backpressure that is added to by inference workers and popped from by train workers. `asyncio.Queue(max.size = )` only operates w/in a single thread.
`await q.put(x)` waits for `< max.size`. `q.put_nowait()` adds immediately and raises if there's an error. `q.get()` waits for an element.

```
c = asyncio.Condition(). async with c: await c.wait_for(bool fn)
```

`c` has a lock that is acquired in the `with c` and released in the `wait_for` moments. We share 1 condition around pausing, weight refresh, stopping. Every change to state notifies all. This is potentially suboptimal but a clean design.

`@ray.remote` turns classes into actors & fns into tasks. When instantiating a class you need to do `MyC.remote(args)` and to execute a task you need `obj_ref = myf.remote(args)`, `res = ray.get(args)`. You can specify the resource requirements in the `@ray.remote(max_retries, num_cpus:...)`. Methods on different actors execute in parallel & methods in the same actor execute serially. `ray.cancel(obj_ref)` cancels a task, but there are really no guarantees

Actors fail if the process dies, or the worker that created the process dies. The owner of an actor is the python process that created it OR the actor that instantiated it as part of its state.


### Little details in our implementation

- The rollout dataclass sets `frozen = True`
- Each rollout needs an id
- When a rollout fails, put it in the queue of completions. The pattern we're going for here is "rollout fails": raise & stop training
- ObjectRefs from Ray tasks are awaitable, but `create_task(x.y.remote())` is incorrect syntax b/c y is not a coroutine.

In the train loop, be sure to use `ray.get()` to wait for tasks that need to be executed synchronously.

## Ray asynchronous training

### One (pretty good) way to do it

Dependencies: asyncio, dataclass, ray

1. We need a fn to load config or parse args (Could be yaml, toml, json etc.) When you have config modify actors using config args, instantiate with `Actor.options(... = config["..."]).remote(...)`

2. Have a dataclass for your rollout, w/ timestamp, weight version, id

3. `InfActor` has an `infer(...)` method and a `set_weights(...)` method. State is weight + version

4. `InfManager` has k inf actors, `asyncio.Queue()` for completed rollouts and checks maximum staleness, max in flight. It also handles weight refresh

5. `async def start(...): asyncio.create_task(self.produce(ind)) for ind in max_in_flight`

6. `self.produce(ind)`: Pick worker based on ind (%). Then in an infinite loop: wait asynch while the paused flag is set or we are stopped. If stopped, return  
Then create a rollout. Once its complete, it `notify_all()` on the class condition variable. This lets a waiting refresh wait until `self.active_rollouts = 0`

7. `async def get_batch(self, bs): await queue.get()`. Append to batch if not stale

8. `async def refresh.weights(self, w)`: wait for rollouts to drain. Set weights on all workers. Update weight version, and `self.paused = False`. `notify_all()`

9. `stop()` `self.stopped = True`, `paused = False`, notify all. cancel all tasks. gather producer tasks, which have been cancelled.

All trainer has to do is start, get batches, & stop.

Our design has a single lock that protects several state fields: `self.stopped`, `self.paused`, `num_active`, rollout id, weight version

Our implementation also means a failed rollout crashes the whole train loop.

If you have a producer complete/submit tasks faster than a consumer can handle them, then you will accumulate task results & eventually OOM. Here's another alternative:

```python
MX_PEND = 100
res_ref = []
for _ in range(NUM_TASKS):  # Very large or infinite
    if len(res_refs) > MX_PEND:
        ready_refs, res_refs = ray.wait(res_refs, num_returns=1)
        # ^ this line splits between completed obj refs & in
        # flight obj refs. It requires len(ready_refs) >= 1 using
        # num_returns, so it'll wait.

        ray.get(ready_refs)  # <- memory freed.

    res_refs.append(actor.big_task.remote())  # Observe
    # only MX_PEND are in flight at once.
```

We don't use `ray.wait` in our implementation, we use `asyncio.Queue`.
We must monitor & balance queue size, inference workers, max staleness, etc.
