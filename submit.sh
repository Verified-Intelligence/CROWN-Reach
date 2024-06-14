docker build . -t "dummy-submit-$1" --no-cache
docker run --name "dummy-submit-container-$1" "dummy-submit-$1" python3 -u /home/run.py
docker cp "dummy-submit-container-$1":/home/CROWN-Reach/results/ .
docker rm --force "dummy-submit-container-$1"
docker image rm --force "dummy-submit-$1"