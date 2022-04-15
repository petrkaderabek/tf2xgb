from setuptools import setup

import tf2xgb


setup(
    name='tf2xgb',
    license='MIT License',
    packages=['tf2xgb'],
    install_requires=['numpy', 'matplotlib', 'pandas', 'sklearn', 'xgboost', 'tensorflow'],
    extras_require={'tests': ['pytest']},
    py_modules=['tf2xgb'],
    version=tf2xgb.__version__,
    description='XGBoost Regression with TensorFlow Pooling and Loss',
    url='https://github.com/petrkaderabek/tf2xgb',
    keywords=['tensorflow', 'xgboost', 'pooling'],
    classifiers=[]
)