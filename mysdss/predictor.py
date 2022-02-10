
import os, random, requests, time, pathlib, base64, tempfile, io, logging
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from ImageCutter.ImageCutter import FITSImageCutter
from astropy.io import fits
import subprocess

logger = logging.getLogger(__name__)

from .helper import Helper

# generic class for predictions
class Predictor:
    def __init__(self, model=None, x=['img', 'spectra', 'bands'], y=['redshift', 'subclass'], helper=None):
        self.x = x
        self.y = y

        if helper is None:
            self.helper = Helper()
        else:
            self.helper = helper

        self.tmpdir = '/tmp/sdss'

        if not os.path.exists(model):
            model = os.path.join(self.helper.FILES, 'models', model)

        if model:
            self.model = tf.keras.models.load_model(model)

        # sky server web server
        self.ssws = 'http://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg'
        self.scale = 0.2
        self.width = 150
        self.height = 150
        self.opt = ''

        pathlib.Path(self.tmpdir).mkdir(parents=True, exist_ok=True)

    def predict(self, objID):
        _row = self.helper.get_row(objID)

        if _row:
            _input, _result = {}, { '_row': _row }
    
            if 'img' in self.x:
                _filename = self._save_image(_row['objID'], _row['ra'], _row['dec'])
                _img, _img_base64 = self._load_img(_filename)
                _input['img'] = np.array([_img])
                _result['_img_base64'] = _img_base64

            if 'fits' in self.x:
                _filename = self._save_fits(_row['objID'])
                _result['_fits_base64'] = []
                with open(_filename, 'rb') as fin:
                    arr = np.load(fin)
                    _input['fits'] = np.array([arr])
                    # FIXME _fits_base64
                    for i in range(5):
                        plt.imsave('/tmp/band.jpg', arr[:, :, i])
                        with open('/tmp/band.jpg', 'rb') as fin:
                            _result['_fits_base64'].append(base64.b64encode(fin.read()).decode('utf-8'))

            if 'spectra' in self.x:
                _filename = self._save_spectra(objID)
                _spectra, _waves = self._load_spectra(_filename)
                _input['spectra'] = np.array([_spectra])
                _result['_waves'] = _waves

            if 'ssel' in self.x:
                _filename = self._save_spectra(objID)
                _ssel, _waves = self._load_ssel(_filename)
                _input['ssel'] = np.array([_ssel])
                _result['_waves'] = _waves

            if 'bands' in self.x:
                # https://skyserver.sdss.org/dr16/SkyServerWS/SearchTools/SqlSearch?cmd=SELECT%20modelMag_u%20AS%20u,modelMag_g%20AS%20g,modelMag_r%20AS%20r,modelMag_i%20AS%20i,modelMag_z%20AS%20z%20FROM%20SpecPhoto%20WHERE%20objID=1237665429165768709&format=csv
                _bands = [_row['u'], _row['g'], _row['r'], _row['i'], _row['z']]
                _input['bands'] = np.array([_bands])

            _output = self.model.predict(_input)

            _result['_classes'] = {}
            for i in self.y:
                if i in CLASSES:
                    _result['_classes'][i] = CLASSES[i][np.argmax(_output[self.y.index(i)])]

            _result['x'] = self.x
            _result['y'] = self.y
            _result['input'] = dict([(x, _input[x].tolist()) for x in _input.keys()])
            if len(self.y) > 1:
                _result['output'] = [x[0].tolist() for x in _output]
            else:
                 _result['output'] = _output[0].tolist()


            return _result
        else:
            return None

    def _load_img(self, _filename):
        _image = tf.keras.preprocessing.image.load_img(_filename)
        _img = tf.keras.preprocessing.image.img_to_array(_image) / 255
        with open(_filename, 'rb') as fin:
            _img_base64 = base64.b64encode(fin.read()).decode('utf-8')

        return _img, _img_base64

    def _save_image(self, objID, ra, dec):
        filename = os.path.join(self.tmpdir, str(objID)+'.jpg')
        
        if os.path.exists(filename):
            return filename

        payload = {
            'ra': ra,
            'dec': dec,
            'scale': self.scale,
            'width': self.width,
            'height': self.height,
            'opt': self.opt
        }
        r = requests.get(self.ssws, params=payload)

        if r.status_code == 200:
            with open(filename, 'wb') as fout:
                fout.write(r.content)

        return filename

    def _load_spectra(self, _filename):
        _df = pd.read_csv(_filename)
        x = _df[(_df['Wavelength']>4000.0) & (_df['Wavelength']<9000.0)]['Flux'].to_numpy()
        _waves = _df[(_df['Wavelength']>4000.0) & (_df['Wavelength']<9000.0)]['Wavelength'].to_list()
        _spectra = x.tolist()

        return _spectra, _waves

    def _load_ssel(self, _filename):
        _df = pd.read_csv(_filename)

        intervals = [(4000,4200),(4452,4474),(4514,4559),(4634,4720),(4800,5134),(5154,5196),(5245,5285),
           (5312,5352),(5387,5415),(5696,5720),(5776,5796),(5876,5909),(5936,5994),(6189,6272),
           (6500,6800),(7000,7300),(7500,7700)]
        dfs = []
        for i in intervals:
            dfs.append(_df[(_df['Wavelength']>=i[0]) & (_df['Wavelength']<=i[1])])

        final = pd.concat(dfs)
        _waves = final['Wavelength'].to_list()
        _spectra = final['BestFit'].to_list()

        return _spectra, _waves

    def _save_spectra(self, objID):
        filename = os.path.join(self.tmpdir, str(objID)+'.csv')

        if os.path.exists(filename):
            return filename

        url = self.helper.spectra_url(objID)
        r = requests.get(url)

        if r.status_code == 200:
            with open(filename, 'wb') as fout:
                fout.write(r.content)
        else:
            print(r.status_code, r.reason)

        return filename

    def _save_fits(self, objID):
        urls_files = self.helper.frames_urls_filenames(objID)
        row = self.helper.get_row(objID)

        # download fits files
        for u, f in urls_files:
            if os.path.exists(f) or os.path.exists(f.replace('.bz2', '')):
                pass
            else:
                r = requests.get(u)
                if r.status_code == 200:
                    with open(f, 'wb') as fout:
                        fout.write(r.content)

        # unzip files
        for _, f in urls_files:
            if os.path.exists(f):
                subprocess.run(['bunzip2', f])

        # build fits data
        _exists = []
        for u, f in urls_files:
            _exists.append(os.path.exists(f.replace('.bz2', '')))

        if all(_exists):
            arr = []
            for _, f in urls_files:
                tmp = tempfile.NamedTemporaryFile()
                
                x = FITSImageCutter()
                x.prepare(f.replace('.bz2', ''))
                x.fits_cut(row['ra'], row['dec'], tmp.name, xs=0.4, ys=0.4)
                
                hdul = fits.open(tmp.name)
                data = hdul[0].data
                if data.shape == (61, 61):
                    arr.append(data)
                else:
                    print('Err shape', _id, f.replace('.bz2', ''))
                
                tmp.close()
            
            if len(arr) == 5:
                with open(self.helper.fits_filename(row['objID']), 'wb') as fout:
                    np.save(fout, np.stack(arr, axis=-1))
            else:
                print('Err len', _id)

        return self.helper.fits_filename(row['objID'])





