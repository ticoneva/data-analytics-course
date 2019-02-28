#!/bin/sh

user=$1
passwd=$2
useradd -m -p `mkpasswd $passwd` $user
cp -a /home/default/. /home/$user/
chown -R $user:$user /home/$user/
chmod 700 /home/$user/

