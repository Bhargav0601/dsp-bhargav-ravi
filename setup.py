from setuptools import setup, find_packages

setup_args = {
    'name': 'house_price_prediction',
    'version': '0.1',
    'packages': find_packages(),
    'install_requires': [
        'flask>=2.0.1',
        'pandas>=1.3.3',
        'scikit-learn>=0.24.2',
        'numpy>=1.21.2',
        'joblib>=1.0.1'
    ],
    'include_package_data': True,
    'package_data': {
        'house_prices': ['../data/*.csv', '../models/*.joblib']
    }
}

if __name__ == '__main__':
    setup(**setup_args)
    
    # Generate requirements.txt automatically
    with open('requirements.txt', 'w') as f:
        f.write('\n'.join(setup_args['install_requires']))