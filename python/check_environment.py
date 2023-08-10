def get_packages(pkgs):
    versions = []
    for p in packages:
        try:
            imported = __import__(p)
            try:
                versions.append(imported.__version__)
            except AttributeError:
                try:
                    versions.append(imported.version)
                except AttributeError:
                    try:
                        versions.append(imported.version_info)
                    except AttributeError:
                        versions.append('0.0')
        except ImportError:
            print('[FAIL]: %s is not installed' % p)
    return versions
                    
packages = ['numpy', 'scipy', 'matplotlib', 'sklearn', 'pandas', 'mlxtend']
suggested_v = ['1.10', '0.17', '1.5.1', '0.17.1', '0.17.1', '0.4.2']
versions = get_packages(packages)

for p, v, s in zip(packages, versions, suggested_v):
    if v < s:
        print('[FAIL] %s %s, please upgrade to >= %s' % (p, v, s))
    else:
        print('[OK] %s %s' % (p, v))

get_ipython().magic('load_ext watermark')
get_ipython().magic('watermark -d -p numpy,scipy,matplotlib,sklearn,pandas,mlxtend')

