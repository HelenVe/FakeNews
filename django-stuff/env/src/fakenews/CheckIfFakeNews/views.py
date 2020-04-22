from django.shortcuts import render
import validators
# Create your views here.

def home(request):
	context={}
	return render(request, "home.html", context) # {}: context variables


def result(request):
	if 'url' in request.GET:
		print("Found url:",request.GET.get('url'))
	else:
		print("SHOULD NEVER REACH THIS. url not in parameters")
		return HttpResponse("<h1>No url specified. No data to display!</h1>",{})

	print(validators.url(request.GET.get('url')))
	url404=validators.url(request.GET.get('url'))

	context={'url':request.GET.get('url'), 'url404':url404}
	return render(request, "result.html", context) # {}: context variables
