import setuptools

setuptools.setup(name='ml-preprocessing',
                 author='Pedro Marques',
                 author_email='pedro.r.marques@gmail.com',
                 description='Dataset preprocessing tools',
                 url='https://github.com/pedro-r-marques/ml-preprocessing',
                 packages=setuptools.find_packages(),
                 install_requires = [
                     'numpy', 'tensorflow', 'pandas'
                 ],
                 python_requires='>=3.6',
                 version='0.0.1')