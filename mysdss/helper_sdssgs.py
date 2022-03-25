
import os, random, logging
import numpy as np
from pandas import read_csv
from tqdm import tqdm
import tensorflow.keras.preprocessing.image as keras

logger = logging.getLogger(__name__)

class Helper():
    def __init__(self, ds='../sdss-gs'):
        self.FILES = ds

        if not os.path.exists(self.FILES):
            logger.warn(f'Dataset files directory not found: { self.FILES }')

        self.df = None
        _filename = os.path.join(self.FILES, 'data.csv')
        if os.path.exists(_filename):
            self.df = read_csv(_filename)
        else:
            logger.warn(f'Data file not found: { _filename }')

    def ids_list(self, has_img=False, has_fits=False, has_spectra=False, has_ssel=False, has_bands=False, has_wise=False, has_gz2c=False):
        _df = self.df.copy()

        if has_bands:
            _df = _df[~_df['modelMag_u'].isna() & ~_df['modelMag_g'].isna() & ~_df['modelMag_r'].isna() & ~_df['modelMag_i'].isna() & ~_df['modelMag_z'].isna()]
        if has_wise:
            _df = _df[~_df['w1mag'].isna() & ~_df['w2mag'].isna() & ~_df['w3mag'].isna() & ~_df['w4mag'].isna()]
        if has_gz2c:
            _df = _df[~_df['gz2c_f'].isna() & ~_df['gz2c_s'].isna()]
        _ids = _df['objid'].tolist()

        if has_img:
            _ids = [x for x in _ids if self._has_img(x)]
        if has_fits:
            _ids = [x for x in _ids if self._has_fits(x)]
        if has_spectra:
            _ids = [x for x in _ids if self._has_spectra(x)]
        if has_ssel:
            _ids = [x for x in _ids if self._has_ssel(x)]

        return _ids

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

    def _has_ssel(self, id):
        return os.path.exists(self.ssel_filename(id))

    def img_filename(self, objID, DIR='img'):
        return os.path.join(self.FILES, DIR, str(objID)+'.jpg')

    def fits_filename(self, objID, DIR='fits'):
        return os.path.join(self.FILES, DIR, str(objID)+'.npy')

    def spectra_filename(self, objID, DIR='spectra'):
        return os.path.join(self.FILES, DIR, str(objID)+'.csv')

    def ssel_filename(self, objID, DIR='ssel'):
        return os.path.join(self.FILES, DIR, str(objID)+'.csv')

    def get_row(self, id):
        if isinstance(id, str):
            id = int(id)

        sl = self.df[self.df['objid'] == id]
        if sl.empty:
            return None
        else:
            return sl.iloc[0].to_dict()

    def y_list(self, ids, target):
        if target in ['redshift', 'stellarmass']:
            res = []
            for i in ids:
                row = self.get_row(i)
                res.append(row[target])

            return res

    def y_list_class(self, ids, target, classes):
        n = len(classes)

        y = []
        for i in ids:
            _row = self.get_row(i)
            if target == 'gz2c':     # exception for gz2class
                c = _row['gz2c_s']
            else:
                c = _row[target]
            tmp = np.zeros(n)
            tmp[classes.index(c)] = 1
            y.append(tmp)

        return y, classes


    # def spectra_url(self, objid):
    #     obj = self.get_obj(objid)

    #     return f"https://dr16.sdss.org/optical/spectrum/view/data/format=csv/spec=lite?plateid={ obj['plate'] }&mjd={ obj['mjd'] }&fiberid={ obj['fiberid'] }"
    #     #return f"https://dr17.sdss.org/optical/spectrum/view/data/format=csv/spec=lite?plateid={ obj['plate'] }&mjd={ obj['mjd'] }&fiberid={ obj['fiberid'] }"

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

    def load_ssels(self, _ids):
        X_ssel = []

        for i in _ids:
            _df = read_csv(self.ssel_filename(i))
            x = _df['BestFit'].to_numpy()
            X_ssel.append(x)

        return np.array(X_ssel)

    def load_bands(self, _ids):
        X_bands = []

        for i in _ids:
            row = self.get_row(i)
            x = [row['modelMag_u'], row['modelMag_g'], row['modelMag_r'], row['modelMag_i'], row['modelMag_z']]
            X_bands.append(x)

        return np.array(X_bands)

    def load_wises(self, _ids):
        X_wise = []

        for i in _ids:
            row = self.get_row(i)
            x = [row['w1mag'], row['w2mag'], row['w3mag'], row['w4mag']]
            X_wise.append(x)

        return np.array(X_wise)


    # def _frame_url(self, obj, band):
    #     return f"https://dr17.sdss.org/sas/dr17/eboss/photoObj/frames/{ obj['rerun'] }/{ obj['run'] }/{ obj['camcol'] }/frame-{ band }-{ str(obj['run']).zfill(6) }-{ obj['camcol'] }-{ str(obj['field']).zfill(4) }.fits.bz2"

    # def _frame_filename(self, obj, band, DIR='sdss-frames-galaxy', bz=False):
    #     d = os.path.join(self.FILES, DIR, str(obj['rerun']), str(obj['run']))
    #     os.makedirs(d, exist_ok=True)
    #     filename = os.path.join(d, str(obj['objid']) + '_' + band + '.fits')
        
    #     if bz:
    #         filename += '.bz2'

    #     return filename

    # def frames_urls_filenames(self, _id):
    #     obj = self.get_obj(_id)

    #     if obj:
    #         r = []
    #         for b in ['u', 'g', 'r', 'i', 'z']:
    #             u = self._frame_url(obj, b)
    #             f = self._frame_filename(obj, b, bz=True)

    #             r.append((u, f))

    #     return r

    def random_id(self):
        _ids = self.df['objid'].tolist()

        return random.choice(_ids)



