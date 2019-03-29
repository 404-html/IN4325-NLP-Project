# IN4325 NLP Project

Welcome to the repository of Group 12!

## Requirements
* Python 3  
* pandas  
* sklearn

## Extracting data from scaledata
Linux command to create `|` seperated `data.csv` file from scaledata which will have a form:
`id|class(0-3)|review_content`
```
paste -d "|" scaledata/Steve+Rhodes/id.Steve+Rhodes \
  		scaledata/Steve+Rhodes/label.3class.Steve+Rhodes \
  		scaledata/Steve+Rhodes/subj.Steve+Rhodes \
        > data.csv
```
        

