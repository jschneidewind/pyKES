# pyKES

Working locally with pyKES:

In other repo running
pip install -e /Users/jacob/Documents/Water_Splitting/Projects/pyKES/pyKES

Adding .vscode/setting.json file with

{
    "python.analysis.extraPaths": [
        "/Users/jacob/Documents/Water_Splitting/Projects/pyKES/pyKES/src"
    ],
    "python.autoComplete.extraPaths": [
        "/Users/jacob/Documents/Water_Splitting/Projects/pyKES/pyKES/src"
    ]
}
 
## Other packages using pyKES

When pyKES version changes:
* Updating packages' pyproject.toml to new pyKES version
running
uv lock --refresh
uv sync



## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).