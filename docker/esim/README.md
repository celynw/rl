eval "$ssh-agent"
ssh-add ~/.ssh/id_rsa
docker build -t esim --ssh default .
docker tag esim celynw/esim
docker login
docker push celynw/esim
