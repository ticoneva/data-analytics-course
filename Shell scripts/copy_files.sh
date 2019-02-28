#!/bin/sh

for userdetails in `cat users.csv`
do
        user=`echo $userdetails | cut -f 1 -d ,`
        cp /home/default/Completed/imdb.npz /home/$user/Empty/
        chown $user:$user /home/$user/Empty/imdb.npz
done
