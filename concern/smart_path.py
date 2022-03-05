from functools import partial
from smart_path import smart_path as spath


smart_path = partial(spath, endpoint_url='http://oss.i.brainpp.cn')