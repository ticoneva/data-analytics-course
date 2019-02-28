#Start temporary notebook containers
#Will appear as http://econ1101.econ.cuhk.edu.hk:8000
#Pool size: 5
#Max Docker Worker: 2

export TOKEN=$( head -c 30 /dev/urandom | xxd -p )
docker run --net=host -d -e CONFIGPROXY_AUTH_TOKEN=$TOKEN --name=proxy ticoneva/configurable-http-proxy --default-target http://127.0.0.1:9999
#docker run --net=host -d -e CONFIGPROXY_AUTH_TOKEN=$TOKEN --name=tmpnb -v /var/run/docker.sock:/docker.sock jupyter/tmpnb

#Start containers
#Resets every 8 hours regardless of activity
docker run -d \
    --net=host \
    -e CONFIGPROXY_AUTH_TOKEN=$TOKEN \
    -v /var/run/docker.sock:/docker.sock \
    jupyter/tmpnb \
    python orchestrate.py \
	--pool-size=5 \
	--cpu-quota=100000 \
	--mem-limit=512m \
	--max-dock-workers=2 \
	--cull-max=28800 --cull-timeout=28800 \
	--image='ticoneva/econ4130-notebook-tf' \
        --command='start-notebook.sh \
            "--NotebookApp.base_url={base_path} \
            --ip=0.0.0.0 \
            --port={port} \
            --NotebookApp.trust_xheaders=True"'
