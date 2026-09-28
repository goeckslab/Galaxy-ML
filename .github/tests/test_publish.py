"""Run with python .github/tests/test_publish.py (standard library only)."""

import io
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from unittest.mock import patch
from urllib.error import HTTPError, URLError


workflow = Path(__file__).parents[1] / 'workflows' / 'publish.yaml'
# Exercise the inline Python used by the workflow, including for old tags
# whose source trees predate this workflow.
source = workflow.read_text()
source = source.split('        shell: python\n        run: |\n')[1]
source = dedent(source.split('\n      - ')[0])
code = compile(source, str(workflow), 'exec')

with TemporaryDirectory(dir='.') as directory:
    output = Path(directory) / 'output'
    summary = Path(directory) / 'summary'
    for tag, response, expected in [
        ('release_v0.11.0', b'{"urls": [{"filename": "old.tar.gz"}]}', False),
        ('release_v0.11.0', b'{"urls": []}', True),
        ('release_v0.11.0', HTTPError('test', 404, '', None, None), True),
        ('release_v0.11.0', HTTPError('test', 500, '', None, None), HTTPError),
        ('release_v0.11.0', URLError('offline'), URLError),
        ('release_v0.11.0', b'not JSON', ValueError),
        ('release_v0.12.0', b'{"urls": []}', SystemExit),
    ]:
        output.write_text('')
        summary.write_text('')
        with (
            patch.dict(os.environ, {'RELEASE_TAG': tag,
                                    'GITHUB_OUTPUT': str(output),
                                    'GITHUB_STEP_SUMMARY': str(summary)}),
            patch('runpy.run_path', return_value={'__version__': '0.11.0'}),
            patch('urllib.request.urlopen') as request,
        ):
            if isinstance(response, Exception):
                request.side_effect = response
            else:
                request.return_value = io.BytesIO(response)
            try:
                exec(code, {})
            except BaseException as error:
                assert isinstance(expected, type), error
                assert isinstance(error, expected), error
                assert output.read_text() == ''
            else:
                assert type(expected) is bool, 'Expected validation to fail'
                expected_output = f'publish={str(expected).lower()}\n'
                assert output.read_text() == expected_output
                skipped = 'skipping upload' in summary.read_text()
                assert skipped == (not expected)
            if tag != 'release_v0.11.0':
                request.assert_not_called()
            else:
                request.assert_called_once_with(
                    'https://pypi.org/pypi/Galaxy-ML/0.11.0/json', timeout=30)

print('Publishing checks passed: existing, missing, empty, mismatched '
      'and failed requests.')
