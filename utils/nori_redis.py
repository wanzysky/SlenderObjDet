import redis
from tqdm import tqdm

import nori2 as nr

from concern.smart_path import smart_path


class NoriRedis(object):
    """Redis connection which holds a hash for a nori dataset.
    The key and value of the hash table are file name and nori data id, respectily.
    NOTICE: NoriRedis is not multiple process/thread safe.
        Thus, initialization should be in sub processes/threads when neccessary.
    """

    def __init__(self, cfg, nori_path):
        self.host = cfg.REDIS.HOST
        self.port = cfg.REDIS.PORT
        self.db = cfg.REDIS.DB
        self.nori_path = nori_path
        self.connected = False

    def connect(self):
        self.connection = redis.Redis(host=self.host, port=self.port, db=self.db)
        self.fechcer = None

    def get(self, file_name):
        if not self.connected:
            self.connect()

        binary = self.connection.hget(self.nori_path, file_name)
        if binary is None:
            raise KeyError("%s:%s"%(self.nori_path, file_name))
        return binary.decode()

    def sync(self):
        if not self.connected:
            self.connect()
        self.connection.delete(self.nori_path)

        with nr.open(self.nori_path) as r:
            for data_id, _, meta in tqdm(r.scan(scan_data=False, scan_meta=True)):
                file_name = meta["filename"]
                if isinstance(file_name, list):
                    file_name = file_name[0]
                self.connection.hset(self.nori_path, file_name, data_id)

    def fetch(self, file_name, fetcher=None):
        data_id = self.get(smart_path(file_name).name)
        if fetcher is None:
            if self.fechcer is None:
                self.fechcer = nr.Fetcher()

            fetcher = self.fechcer

        return fetcher.get(data_id)
