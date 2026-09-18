# Publishing as a public GitHub repository

This tree is a self-contained project (code, data, fitted model, paper, tests).
It was prepared for **https://github.com/kelvinvalani/vic-rent-ml**. Creating that
repository requires a GitHub token with the “create repository” permission, which
the automated run did not have.

To publish from this checkout after you create an empty **public** GitHub repo
named `vic-rent-ml` (no README, no `.gitignore`, no licence):

```sh
git checkout -b main
git remote add public https://github.com/kelvinvalani/vic-rent-ml.git
git push -u public main
```

Then add topics such as `machine-learning`, `forecasting`, `victoria`, `rental-market`
on the GitHub settings page if you want them.
