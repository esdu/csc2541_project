# This file collects the bandit code and puts it here

pushd ~
rm -rf RecommendationSystems # Remove any existing stuff
git clone https://github.com/scheeloong/RecommendationSystems
cd RecommendationSystems
bash install.sh
popd 

python3 banditApiTest.py
