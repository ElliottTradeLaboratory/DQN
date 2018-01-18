import sys
import os
import re

for path in sys.path:
    if re.match(r'^.*site-packages/alewrap_py-[\d]\.[\d]\.[\d]-py[\d]\.[\d]\.egg$', path):
        print(re.sub(r'alewrap_py-[\d]\.[\d]\.[\d]-py[\d]\.[\d]\.egg', '', path))
        sys.exit(0)
print('you have not install alewrap_py')
sys.exit(1)
