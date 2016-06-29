from pprint import pprint
import todoist

api = todoist.TodoistAPI('b4eb1878ccf268e65d052d18fd3961a9aca46e75')
for project in api.sync(resource_types=['all'])['Projects']:
    pprint(project['name'])
