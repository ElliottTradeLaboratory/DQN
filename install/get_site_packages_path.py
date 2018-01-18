import sys
import os
import re

try:
    get_alewrap_dir = sys.argv.index('--get_alewrap_dir')
except ValueError:
    get_alewrap_dir = False

for path in sys.path:
    if re.match(r'^.*(site|dist)-packages/alewrap_py-[\d]\.[\d]\.[\d]-py[\d]\.[\d]\.egg$', path):
        if get_alewrap_dir:
            print(path)
        else:
            print(re.sub(r'alewrap_py-[\d]\.[\d]\.[\d]-py[\d]\.[\d]\.egg', '', path))
        sys.exit(0)
print('you have not install alewrap_py')
sys.exit(1)
