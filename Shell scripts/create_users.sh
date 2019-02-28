#!/bin/sh

for userdetails in `cat users_2.csv`
do
        user=`echo $userdetails | cut -f 1 -d ,`
        passwd=`echo $userdetails | cut -f 2 -d ,`
        useradd -m -p `mkpasswd $passwd` $user
        cp -a /home/default/. /home/$user/
        chown -R $user:$user /home/$user/
        chmod 700 /home/$user/
        #usermod -aG $user teacher
        #usermod -aG $user ta
done
