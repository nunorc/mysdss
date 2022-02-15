
import logging, os, pickle

logger = logging.getLogger(__name__)

class Cache:
    def __init__(self, path='./.mysdss_cache'):
        self.path = path

        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)

    def _filename(self, key):
        return os.path.join(self.path, f'{ key }.pkl')

    def get(self, key):
        logger.debug(f'Cache get: { key }')
        f = self._filename(key)

        if os.path.exists(f):
            with open(f, 'rb') as fin:
                return pickle.load(fin)
        else:
            return None

    def set(self, key, data):
        logger.debug(f'Cache set: { key }')
        with open(self._filename(key), 'wb') as fout:
            pickle.dump(data, fout)

