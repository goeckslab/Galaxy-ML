import argparse
import json
import numpy as np
import plotly
import plotly.graph_objs as go
import pandas as pd
import warnings

from sklearn.feature_selection.base import SelectorMixin
from sklearn.pipeline import Pipeline
from galaxy_ml.utils import load_model


def main(inputs, infile_estimator=None, infile1=None,
         infile2=None, outfile_result=None,
         outfile_object=None, groups=None,
         ref_seq=None, intervals=None,
         targets=None, fasta_path=None):
    """
    Parameter
    ---------
    inputs : str
        File path to galaxy tool parameter

    infile_estimator : str, default is None
        File path to estimator

    infile1 : str, default is None
        File path to dataset containing features

    infile2 : str, default is None
        File path to dataset containing target values

    outfile_result : str, default is None
        File path to save the results, either cv_results or test result

    outfile_object : str, default is None
        File path to save searchCV object

    groups : str, default is None
        File path to dataset containing groups labels

    ref_seq : str, default is None
        File path to dataset containing genome sequence file

    intervals : str, default is None
        File path to dataset containing interval file

    targets : str, default is None
        File path to dataset compressed target bed file

    fasta_path : str, default is None
        File path to dataset containing fasta file
    """
    warnings.simplefilter('ignore')

    with open(inputs, 'r') as param_handler:
        params = json.load(param_handler)
    
    if (params['plotting_selection']
              ['plot_type']) == 'feature_importances':
        with open(infile_estimator, 'rb') as estimator_handler:
            estimator = load_model(estimator_handler)

        df = pd.read_csv(infile1, sep='\t', header='infer', index_col=None)
        feature_names = df.columns.values

        if isinstance(estimator, Pipeline):
            for st in estimator.steps[:-1]:
                if isinstance(st[-1], SelectorMixin):
                    mask = st[-1].get_support()
                    feature_names = feature_names[mask]
            estimator = estimator.steps[-1][-1]
        
        if hasattr(estimator, 'coef_'):
            coefs = estimator.coef_
        else:
            coefs = getattr(estimator, 'feature_importances_', None)
        if coefs is None:
            raise RuntimeError('The classifier does not expose '
                               '"coef_" or "feature_importances_" '
                               'attributes')

        threshold = params['plotting_selection']['threshold']
        if threshold is not None:
            coefs = coefs[coefs >= threshold]
            feature_names = feature_names[coefs >= threshold]
        
        # sort
        indices = np.argsort(coefs)[::-1]

        trace = go.Bar(x=feature_names[indices],
                       y=coefs[indices]) 
        layout = go.Layout(title="Feature importances")
        plotly.offline.plot([trace], filename = "output.html",
                            auto_open=False)


if __name__ == '__main__':
    aparser = argparse.ArgumentParser()
    aparser.add_argument("-i", "--inputs", dest="inputs", required=True)
    aparser.add_argument("-e", "--estimator", dest="infile_estimator")
    aparser.add_argument("-X", "--infile1", dest="infile1")
    aparser.add_argument("-y", "--infile2", dest="infile2")
    aparser.add_argument("-O", "--outfile_result", dest="outfile_result")
    aparser.add_argument("-o", "--outfile_object", dest="outfile_object")
    aparser.add_argument("-g", "--groups", dest="groups")
    aparser.add_argument("-r", "--ref_seq", dest="ref_seq")
    aparser.add_argument("-b", "--intervals", dest="intervals")
    aparser.add_argument("-t", "--targets", dest="targets")
    aparser.add_argument("-f", "--fasta_path", dest="fasta_path")
    args = aparser.parse_args()

    main(args.inputs, args.infile_estimator, args.infile1, args.infile2,
         args.outfile_result, outfile_object=args.outfile_object,
         groups=args.groups, ref_seq=args.ref_seq, intervals=args.intervals,
         targets=args.targets, fasta_path=args.fasta_path)
