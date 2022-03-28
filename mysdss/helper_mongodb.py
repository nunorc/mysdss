
import os, random, logging
import numpy as np
from pandas import read_csv
from tqdm import tqdm
import tensorflow.keras.preprocessing.image as keras
import pymongo

logger = logging.getLogger(__name__)

from .cache import Cache

def _guess_files(FILES):
    if os.path.exists(FILES):
        return FILES
    tmp = os.path.join('.', FILES.split('/')[-1])
    if os.path.exists(tmp):
        return tmp
    tmp = os.path.join('..', FILES.split('/')[-1])
    if os.path.exists(tmp):
        return tmp
    
    logger.warning('FILES not found: '+FILES)
    return None


class Helper():
    def __init__(self, FILES='../../FILES', mongodb='mongodb://localhost:27017/', col='sdss', cache=True):
        self.FILES = _guess_files(FILES)
        
        if mongodb:
            self.client = pymongo.MongoClient(mongodb)
            self.db = self.client['astro']
            self.col = self.db[col]

        if cache:
            self.cache = Cache()
        else:
            self.cache = None

    def ids_list(self, has_img=False, has_fits=False, has_spectra=False, has_ssel=False, has_bands=False, has_wise=False, has_gz2c=False):
        if self.cache:
            key = self._ids_list_key(has_img, has_fits, has_spectra, has_ssel, has_bands, has_wise, has_gz2c)
            data = self.cache.get(key)

            if data:
                return data

        q = {}

        if has_bands:
            q['bands'] = { '$exists': 1 }
        if has_wise:
            q['wise'] = { '$exists': 1 }
        if has_gz2c:
            q['gz2class'] = { '$exists': 1 }

        _ids = [x['_id'] for x in self.col.find(q, { '_id': 1 })]

        if has_img:
            _ids = [x for x in _ids if self._has_img(x)]
        if has_fits:
            _ids = [x for x in _ids if self._has_fits(x)]
        if has_spectra:
            _ids = [x for x in _ids if self._has_spectra(x)]
        if has_ssel:
            _ids = [x for x in _ids if self._has_ssel(x)]

        if self.cache:
            self.cache.set(key, _ids)

        return _ids

    def _ids_list_key(self, *args):
        return f'ids_list_{ self.col.name }_' + "_".join([str(x) for x in args])

    def _has_img(self, _id):
        return os.path.exists(self.img_filename(_id))

    def _has_fits(self, _id):
        return os.path.exists(self.fits_filename(_id))

    def _has_spectra(self, _id):
        filename = self.spectra_filename(_id)
        if os.path.exists(filename):
            _df = read_csv(filename)
            if len(_df)>0 and 'Wavelength' in _df.columns and 'BestFit' in _df.columns:
                _x = _df[(_df['Wavelength']>=4000) & (_df['Wavelength']<=9000.0)]['BestFit'].to_numpy()
                if len(_x) == 3522:
                    return True
        else:
            return False

    def y_list(self, ids, target):
        if target in ['redshift', 'stellarmass']:
            res = []
            for i in ids:
                d = self.col.find_one({ '_id': i }, { target: 1 })
                res.append(d[target])

            return res

    def y_list_class(self, ids, target, classes):
        n = len(classes)

        y = []
        for i in ids:
            o = self.get_obj(i)
            if target == 'gz2c':     # exception for gz2class
                c = o['gz2class']['s']
            else:
                c = o[target]
            tmp = np.zeros(n)
            tmp[classes.index(c)] = 1
            y.append(tmp)

        return y, classes

    def get_obj(self, _id):
        if isinstance(_id, str):
            _id = int(_id)

        return self.col.find_one({ '_id': _id })

    def img_filename(self, objID, DIR='sdss-img-galaxy'):
        d = str(objID)[-1]
        os.makedirs(os.path.join(self.FILES, DIR, d), exist_ok=True)
        filename = os.path.join(self.FILES, DIR, d, str(objID)+'.jpg')
        
        return filename

    def spectra_filename(self, objID, DIR='sdss-spectra-galaxy'):
        d = str(objID)[-1]
        os.makedirs(os.path.join(self.FILES, DIR, d), exist_ok=True)
        filename = os.path.join(self.FILES, DIR, d, str(objID)+'.csv')
        
        return filename

    def spectra_url(self, objid):
        obj = self.get_obj(objid)

        return f"https://dr16.sdss.org/optical/spectrum/view/data/format=csv/spec=lite?plateid={ obj['plate'] }&mjd={ obj['mjd'] }&fiberid={ obj['fiberid'] }"
        #return f"https://dr17.sdss.org/optical/spectrum/view/data/format=csv/spec=lite?plateid={ obj['plate'] }&mjd={ obj['mjd'] }&fiberid={ obj['fiberid'] }"

    def load_imgs(self, _ids):
        X_img = []

        for i in _ids:
            filename = self.img_filename(i)
            img = keras.load_img(filename)
            x = keras.img_to_array(img)/255
            X_img.append(x)

        return np.array(X_img)

    def load_fits(self, _ids):
        X_fits = []

        for i in _ids:
            filename = self.fits_filename(i)
            with open(filename, 'rb') as fin:
                X_fits.append(np.load(fin))

        return np.array(X_fits)

    def load_spectras(self, _ids):
        X_spectra = []

        for i in _ids:
            _df = read_csv(self.spectra_filename(i))
            x = _df[(_df['Wavelength']>4000.0) & (_df['Wavelength']<9000.0)]['Flux'].to_numpy()
            X_spectra.append(x)

        return np.array(X_spectra)

    def load_bands(self, _ids):
        X_bands = []

        for i in _ids:
            o = self.get_obj(i)
            X_bands.append(o['bands'])

        return np.array(X_bands)

    def load_wises(self, _ids):
        X_wise = []

        for i in _ids:
            o = self.get_obj(i)
            X_wise.append(o['wise'])

        return np.array(X_wise)

    def _frame_url(self, obj, band):
        return f"https://dr17.sdss.org/sas/dr17/eboss/photoObj/frames/{ obj['rerun'] }/{ obj['run'] }/{ obj['camcol'] }/frame-{ band }-{ str(obj['run']).zfill(6) }-{ obj['camcol'] }-{ str(obj['field']).zfill(4) }.fits.bz2"

    def _frame_filename(self, obj, band, DIR='sdss-frames-galaxy', bz=False):
        d = os.path.join(self.FILES, DIR, str(obj['rerun']), str(obj['run']))
        os.makedirs(d, exist_ok=True)
        filename = os.path.join(d, str(obj['objid']) + '_' + band + '.fits')
        
        if bz:
            filename += '.bz2'

        return filename

    def frames_urls_filenames(self, _id):
        obj = self.get_obj(_id)

        if obj:
            r = []
            for b in ['u', 'g', 'r', 'i', 'z']:
                u = self._frame_url(obj, b)
                f = self._frame_filename(obj, b, bz=True)

                r.append((u, f))

        return r

    def fits_filename(self, objID, DIR='sdss-fits-galaxy'):
        d = str(objID)[-1]
        os.makedirs(os.path.join(self.FILES, DIR, d), exist_ok=True)
        filename = os.path.join(self.FILES, DIR, d, str(objID)+'.npy')
        
        return filename

    def random_id(self):
        _ids = [x['_id'] for x in self.col.find({}, { '_id': 1 })]

        return random.choice(_ids)

    def ssel_filename(self, objID, DIR='sdss-ssel-galaxy'):
        d = str(objID)[-1]
        os.makedirs(os.path.join(self.FILES, DIR, d), exist_ok=True)
        filename = os.path.join(self.FILES, DIR, d, str(objID)+'.csv')

        return filename

    def _has_ssel(self, id):
        return os.path.exists(self.ssel_filename(id))

    def load_ssels(self, _ids):
        X_ssel = []

        for i in _ids:
            _df = read_csv(self.ssel_filename(i))
            x = _df['BestFit'].to_numpy()
            X_ssel.append(x)

        return np.array(X_ssel)


