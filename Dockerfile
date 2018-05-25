FROM jupyter/tensorflow-notebook

MAINTAINER "Vinci Chow <ticoneva@gmail.com>"

USER root

#Jupyerhub other stuff
ADD . /src/jupyterhub
WORKDIR /src/jupyterhub

RUN mkdir -p /srv/jupyterhub/
WORKDIR /srv/jupyterhub/
EXPOSE 8000

LABEL org.jupyter.service="jupyterhub"

CMD ["jupyterhub"]
