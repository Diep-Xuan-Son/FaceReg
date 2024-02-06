import os

import redis
from rq import Worker, Queue, Connection
import multiprocessing
# import queue

# os.system("service redis-server start")

listen = ['default']

redis_url = os.getenv('REDISTOGO_URL', 'redis://localhost:6380')

conn = redis.from_url(redis_url)

# q_output = queue.Queue(maxsize=1)
# q_com = queue.Queue(maxsize=1)
q = Queue(connection=conn)
def start_worker():
    Worker([q], connection=conn).work()

if __name__ == '__main__':
    with Connection(conn):
        #worker = Worker(list(map(Queue, listen)))
        #worker.work()

        # NUM_WORKERS = multiprocessing.cpu_count()
        NUM_WORKERS = 1
        print("NUM_WORKERS: ", NUM_WORKERS)
        procs = []
        for i in range(NUM_WORKERS):
            proc = multiprocessing.Process(target=start_worker)
            procs.append(proc)
            proc.start()
        print(procs)
