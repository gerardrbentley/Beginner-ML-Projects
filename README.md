# Beginner AI

Streamlit App Boilerplate that contains everything you need to get running.

Built with ❤️ by [Gerard Bentley](https://github.com/gerardrbentley)

<a href="https://gitpod.io/#https://github.com/gerardrbentley/Beginner-ML-Projects" rel="nofollow noopener noreferrer" target="_blank" class="after:hidden"><img src="https://gitpod.io/button/open-in-gitpod.svg" alt="Open in Gitpod"></a>

Click the gitpod link above to set up your own development environment in the cloud without downloading or installing anything!
(See below for local setup instructions)

## What's this?

- `README.md`: This Document! To help you find your way around
- `streamlit_app.py`: The main app that gets run by [`streamlit`](https://docs.streamlit.io/)
- `requirements.txt`: Pins the version of packages needed
- `LICENSE`: Follows Streamlit's use of Apache 2.0 Open Source License
- `.gitignore`: Tells git to avoid comitting / scanning certain local-specific files
- `.streamlit/config.toml`: Customizes the behaviour of streamlit without specifying command line arguments (`streamlit config show`)
- `.gitpod.yml`: Special file to tell Gitpod how to start running your app (see [docs](https://www.gitpod.io/docs/config-gitpod-file) and [python docs](https://www.gitpod.io/docs/languages/python))

## Local Setup

Assumes working python installation and some command line knowledge ([install python with conda guide](https://tech.gerardbentley.com/python/beginner/2022/01/29/install-python.html)).

```sh
# External users: download Files
git clone git@github.com:gerardrbentley/Beginner-ML-Projects.git
# if you don't have ssh set up use the following
# git clone https://github.com/gerardrbentley/Beginner-ML-Projects.git

# Go to correct directory
cd Beginner-ML-Projects

# Create virtual environment for this project
python -m venv venv

# Activate the virtual environment
. ./venv/bin/activate
# .\venv\Scripts\activate for Windows

# Install required Packages
python -m pip install -r requirements.txt

# Run the streamlit app
streamlit run streamlit_app.py
```

Open your browser to [http://localhost:8501/](http://localhost:8501/) if it doesn't open automatically.

## Deploy

For the easiest experience, deploy to [Streamlit Cloud](https://streamlit.io/cloud)

For other options, see [Streamilt deployment wiki](https://discuss.streamlit.io/t/streamlit-deployment-guide-wiki/5099)

## Credits

This package was created with Cookiecutter and the `gerardrbentley/cookiecutter-streamlit` project template.

- Cookiecutter: [https://github.com/audreyr/cookiecutter](https://github.com/audreyr/cookiecutter)
- `gerardrbentley/cookiecutter-streamlit`: [https://github.com/gerardrbentley/cookiecutter-streamlit](https://github.com/gerardrbentley/cookiecutter-streamlit)
