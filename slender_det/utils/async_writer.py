from typing import Union
from multiprocessing import Queue, Process

import nori2 as nr

from concern.smart_path import smart_path


class AsyncWriter(object):
    """An async file writer. It contineously writes content to the path.
    """
    def __init__(
            self,
            path:str,
            queue:Queue=None,
            mode="wb",
            after=None):
        assert isinstance(path, str), "Only string path for process safety."

        self.path = path
        self.mode = mode
        self.queue = Queue(maxsize=32) if queue is None else queue
        self.process = Process(target=self.target, args=(self.queue,))
        self.writer = None
        self.after = after

    def start(self):
        self.process.start()
        return self

    def join(self):
        self.process.join()

    def target(self, queue):
        path = smart_path(self.path)
        with nr.open(path.as_uri(), "w") as writer:
            while True:
                content = queue.get()
                if content is None:
                    break
                writer.put(content[0], filename=[content[1]])
                if self.after is not None:
                    self.after()

    def clear(self):
        self.writer.close()

    def write(self, data, filename):
        self.writer.put(data, filename=filename)

