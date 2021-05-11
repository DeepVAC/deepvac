import base64
import hashlib
import os
import sys
import string
import random
from .log import LOG

try:
    from Crypto import Random
    from Crypto.Cipher import AES
except:
    LOG.logE("Crypto package missing, you must install it with [pip3 install pycrypto]", exit=True)

class SyszuxCipher(object):
    def __init__(self, key):
        self.aes_bs = AES.block_size
        LOG.logI('Using block size = {}'.format(self.aes_bs))
        self.key = hashlib.sha256(key.encode()).digest()
        LOG.logI('Hash of key={} is {}'.format(key, self.key))

    def encrypt(self, pt_byte):
        pt_byte = self._pad(pt_byte)
        iv = Random.new().read(AES.block_size)
        LOG.logI('iv: {}'.format(iv))
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return iv + cipher.encrypt(pt_byte)

    def decrypt(self, pt_byte):
        iv = pt_byte[:self.aes_bs]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpad(cipher.decrypt(pt_byte[self.aes_bs:]))#.decode('utf-8')

    def _pad(self, s):
        return s + str.encode((self.aes_bs - len(s) % self.aes_bs) * chr(self.aes_bs - len(s) % self.aes_bs))

    def __call__(self, path, mode='encrypt'):
        filename, file_extension = os.path.splitext(path)

        if mode == 'encrypt':
            output_filename = "{}_syszux.so".format(filename)
        else:
            output_filename = "{}.{}".format(filename, 'pt')

        if not os.path.exists(path):
            LOG.logE('File does not exist at path: %s'.format(path), exit=True)

        pt_byte = None
        with open(path, 'rb') as f:
            pt_byte = f.read()

        if mode == 'encrypt':
            pt_byte = self.encrypt(pt_byte)
        else:
            pt_byte = self.decrypt(pt_byte)

        with open(output_filename, 'wb') as f:
            f.write(pt_byte)

        LOG.logI('Saved with key={} to {}'.format(self.key, output_filename))

    @staticmethod
    def _unpad(s):
        return s[:-ord(s[-1:])]

if __name__ == "__main__":
    modes = ['encrypt', 'decrypt']
    if len(sys.argv) < 3:
        LOG.logE('Usage: {} <file_path> <mode> <key>'.format(sys.argv[0]), exit=True)

    file = sys.argv[1]
    mode = sys.argv[2]
    if mode not in modes:
        LOG.logE("Unsupported mode: {}".format(mode), exit=True)
 
    key = sys.argv[3]
    sc = SyszuxCipher(key)

    LOG.logI("You will use {} to {} {}".format(key, mode, file))
    sc(file, mode)
