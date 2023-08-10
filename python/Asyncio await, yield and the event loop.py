import asyncio


async def foo():
    print('Executing foo, sleeping...')
    await asyncio.sleep(0)
    print('...foo back from sleeping')
    return 1


async def bar():
    print('Executing bar')
    print('>> bar calling baz, I expected a context switch to foo, instead...')
    baz_result = await baz()
    print('<< bar back from calling baz')
    return baz_result


async def baz():
    print('>>> Executing baz, sleeping, the context switch to foo happens now?')
    await asyncio.sleep(0)
    print('<<< ...baz back from sleeping')
    return 2

loop = asyncio.get_event_loop()
grouped_task = asyncio.gather(foo(), bar())

results = loop.run_until_complete(grouped_task)
print("The results are: {}".format(results))

async def bar_():
    print('Executing bar_')
    print('>> bar_ calling baz_, I expected a context switch to foo, and indeed...')
    baz_result = await baz_()
    print('<< bar_ back from calling baz_')
    return baz_result

@asyncio.coroutine
def baz_():
    yield  # yields control to the event loop, there's no equivalent for await
    print('>>> Executing baz_, sleeping...')
    yield from asyncio.sleep(0)
    print('<<< ...baz_ back from sleeping')
    return 2

loop = asyncio.get_event_loop()
grouped_task = asyncio.gather(foo(), bar_())

results = loop.run_until_complete(grouped_task)
print("The results are: {}".format(results))

