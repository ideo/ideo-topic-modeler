# IDEO Topic Modeler

A IDEO maintained package for all your topic modeling needs!


### Development

We use `pipenv` to manage dependencies.
```bash
pipenv install
```

#### Packaging Up

The `setup.py` file is manually maintained. This reposity can be installed as an editable package into other projects, via ssh, with:
```bash
pipenv install git+https://github.com/ideo/ideo-topic-modeler.git@main#egg=ideo_topic_modeler
```