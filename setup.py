from setuptools import setup


setup(
    name='sn_studies',
    version='0.1',
    description='Studies for supernovae',
    url='http://github.com/lsstdesc/sn_studies',
    author='Philippe Gris',
    author_email='philippe.gris@clermont.in2p3.fr',
    license='BSD',
    packages=['sn_design_dd_survey','sn_saturation'],
    python_requires='>=3.5',
    zip_safe=False,
    install_requires=[
        'sn_tools>=0.1',
    ],
)
