import validators

theUrl="http://google.com"
if "https://" not in theUrl and "http://" not in theUrl:
	theUrl = "https://" + theUrl
print("the url:", theUrl)
print(validators.url(theUrl))