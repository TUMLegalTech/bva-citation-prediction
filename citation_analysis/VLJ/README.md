# Citation Analysis

The latest regexes used for citation extraction (as in db.citations) are:
```python
case_patt = re.compile('(See\s|See,\se\.g\.,\s|See\salso|)([DeMcO\'Mac]*[A-Z][a-z]+\s[A-Z]?[a-z]*\s?v\.\s[DeMcO\'Mac]*[A-Z][a-z]+)([,\s]+|)(\d{1,4})\s(F.2d|F.3d|F.|Vet. App.|U.S.|F. Supp.|F.Supp.|F. Supp. 2d|F. Supp. 3d|F.Supp.2d|F.Supp.3d|F. Supp. 2d.|F. Supp. 3d.)(\s|\sat\s)(\d{1,4})[,\s]*(\d{1,4}[,\-\s]*\d*|)[,\-\s]*(\d*|)(\s?\([A-Za-z\.\s]*\d{4}\)|)')
statute_patt = re.compile('([0-9]{1,4}\s)(C.F.R.|U.S.C.A.|U.S.C.|C. F. R.|U. S. C.| U. S. C. A.)\s[\\xa7]*\s*(\d{1,4}[\.,\s]*\d{1,4}[\(\)A-Za-z\d]*)([,\s\d]*)(\([A-Za-z\.\s]*\d{4}\)|)')
```

In reverse chronological order:

v4 update (04/16/2020 & 04/23/2020):

- VLJ performance analysis
  - Citation rate drop before/after intervention
  - Citation rate trend outlier detection
- Incorporate VLJ metadata

v3 update (04/09/2020):

- Re-extracted all citations after improving regex match and format
- Extract VLJ names from decision document

v2 update (03/28/2020):

- Extracted citations for all documents and inserted into a `citations` collection. The former `single_citations` collection is dropped.
- Plots regenerated to focus on appeal eligible cases and incorporate favorable/unfavorable CAVC dispositions

v1 update:

- Extract citations for single issue cases. which are loaded into a `single_citations` collection
- Preliminary plots on appeal rate, CAVC disposition distribution, and CAVC remand rate

