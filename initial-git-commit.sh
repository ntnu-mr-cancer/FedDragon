git init
git add --chmod=+x -- build.sh export.sh train-test.sh
git add -A
git commit -m "Initial release"
git remote add origin https://github.com/DIAGNijmegen/dragon_roberta_large_domain_specific_v2
git push -u origin main
gh repo edit https://github.com/DIAGNijmegen/dragon_roberta_large_domain_specific_v2 --description "DRAGON RoBERTa Large Domain-specific V2"
