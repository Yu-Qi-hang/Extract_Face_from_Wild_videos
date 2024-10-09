for language in "$@"
do
python facequality.py --language $language
done
