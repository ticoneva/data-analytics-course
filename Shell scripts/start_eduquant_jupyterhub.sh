docker run -p 8000:8000 -d -v ~/ssl/:/etc/ssl/eduquant/ --name data-science-jupyterhub ticoneva\data-science-jupyterhub  jupyterhub --ssl-key /etc/ssl/eduquant/eduquant.key --ssl-cert /etc/ssl/eduquant/eduquant.crt

docker run -p 8102:8000 -d -v ~/ssl/:/etc/ssl/eduquant/ --name econ4130-jupyterhub-ssl econ4130-jupyterhub  jupyterhub --ssl-key /etc/ssl/eduquant/eduquant.key --ssl-cert /etc/ssl/eduquant/eduquant.crt

#sudo service nvidia-docker start