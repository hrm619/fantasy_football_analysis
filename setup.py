from setuptools import setup, find_packages

setup(
    name="fantasy_football_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.12.0",
        "pyyaml>=6.0.0",
        "plotly>=5.18.0",
        "dash>=2.13.0",
        "dash-bootstrap-components>=1.5.0",
    ],
    python_requires=">=3.9",
    author="hm",
    author_email="hrm619@gmail.com",
    description="Fantasy Football Analysis of 2024 Season",
    keywords="fantasy, football, analysis, data science",
    entry_points={
        "console_scripts": [
            "run-dashboard=dashboard.app:main",
        ],
    },
)
