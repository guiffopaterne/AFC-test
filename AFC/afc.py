"""Correspondence Analysis (CA)"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import base
from sklearn import utils

from . import plot
from . import util
from . import svd


class CA(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, n_components=2, n_iter=10, copy=True, check_input=True,
                 random_state=None, engine='auto'):
        self.n_components = n_components
        self.n_iter = n_iter
        self.copy = copy
        self.check_input = check_input
        self.random_state = random_state
        self.engine = engine

    def fit(self, X, y=None):

        # Check input
        if self.check_input:
            utils.check_array(X)

        # Check all values are positive
        if (X < 0).any().any():
            raise ValueError("All values in X should be positive")

        _, row_names, _, col_names = util.make_labels_and_names(X)

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        if self.copy:
            X = np.copy(X)

        X = X.astype(float) / np.sum(X)

        # calcules des marges
        self.row_masses_ = pd.Series(X.sum(axis=1), index=row_names)
        self.col_masses_ = pd.Series(X.sum(axis=0), index=col_names)

        # Standardisation 
        r = self.row_masses_.to_numpy()
        c = self.col_masses_.to_numpy()
        S = sparse.diags(r ** -.5) @ (X - np.outer(r, c)) @ sparse.diags(c ** -.5)
        # Utilisation du SVD pour la standardisation des donnees
        self.U_, self.s_, self.V_ = svd.compute_svd(
            X=S,
            n_components=self.n_components,
            n_iter=self.n_iter,
            random_state=self.random_state,
            engine=self.engine
        )

        # Total des donnees exprimer
        self.total_inertia_ = np.einsum('ij,ji->', S, S.T)

        return self

    def _check_is_fitted(self):
        utils.validation.check_is_fitted(self, 'total_inertia_')

    def transform(self, X):
        self._check_is_fitted()
        if self.check_input:
            utils.check_array(X)
        return self.row_coordinates(X)

    @property
    def eigenvalues_(self):
        """valeurs propres"""
        self._check_is_fitted()

        return np.square(self.s_).tolist()

    @property
    def explained_inertia_(self):
        """Pourcentage d'inertie."""
        self._check_is_fitted()
        return [eig / self.total_inertia_ for eig in self.eigenvalues_]

    @property
    def F(self):
        self._check_is_fitted()
        return pd.DataFrame(np.diag(self.row_masses_ ** -0.5) @ self.U_@ np.diag(self.s_),
                            index=self.row_masses_.index, columns=pd.RangeIndex(0, len(self.s_)))

    @property
    def G(self):
        return pd.DataFrame(np.diag(self.col_masses_ ** -0.5) @ self.V_.T @ np.diag(self.s_),
                            index=self.col_masses_.index, columns=pd.RangeIndex(0, len(self.s_)))


    def row_coordinates(self, X):
        """composante principale ligne."""
        self._check_is_fitted()

        _, row_names, _, _ = util.make_labels_and_names(X)

        if isinstance(X, pd.DataFrame):
            try:
                X = X.sparse.to_coo().astype(float)
            except AttributeError:
                X = X.to_numpy()

        if self.copy:
            X = X.copy()
        # Normalisation des donnees
        if isinstance(X, np.ndarray):
            X = X / X.sum(axis=1)[:, None]
        else:
            X = X / X.sum(axis=1)

        return pd.DataFrame(
            data=X @ sparse.diags(self.col_masses_.to_numpy() ** -0.5) @ self.V_.T,
            index=row_names
        )

    def column_coordinates(self, X):
    #    coordonneer principal pour les colonnes
        self._check_is_fitted()

        _, _, _, col_names = util.make_labels_and_names(X)

        if isinstance(X, pd.DataFrame):
            is_sparse = X.dtypes.apply(pd.api.types.is_sparse).all()
            if is_sparse:
                X = X.sparse.to_coo()
            else:
                X = X.to_numpy()
        if self.copy:
            X = X.copy()
        if isinstance(X, np.ndarray):
            X = X.T / X.T.sum(axis=1)[:, None]
        else:
            X = X.T / X.T.sum(axis=1)

        return pd.DataFrame(
            data=X @ sparse.diags(self.row_masses_.to_numpy() ** -0.5) @ self.U_,
            index=col_names
        )

    def row_contributions(self):
       
    #    contribution des lignes
        F = self.F
        cont_r = (np.diag(self.row_masses_) @ (F**2)).div(self.s_**2)
        return pd.DataFrame(cont_r.values, index=self.row_masses_.index)
    
    def column_contributions(self):
        # contribution des colonne
        G = self.G
        cont_c = (np.diag(self.col_masses_) @ (G**2)).div(self.s_**2)
        return pd.DataFrame(cont_c.values, index=self.col_masses_.index)

    def row_cos2(self):
        F = self.F
        return (F**2).div(np.diag(F @ F.T)**2, axis=0)

    def column_cos2(self):
        G = self.G
        return (G**2).div(np.diag(G @ G.T)**2, axis=0)


    def plot_coordinates(self, X, ax=None, figsize=(6, 6), x_component=0, y_component=1,
                                   show_row_labels=True, show_col_labels=True, **kwargs):

        self._check_is_fitted()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # AJouter les style pour avoir les graphes beautiful
        ax = plot.stylize_axis(ax)

        # recuperer les libelle ligne et colonne du datasets
        row_label, row_names, col_label, col_names = util.make_labels_and_names(X)

        # Plot ligne des coordonnees principales
        row_coords = self.row_coordinates(X)
        ax.scatter(
            row_coords[x_component],
            row_coords[y_component],
            **kwargs,
            label=row_label
        )

        # affichages des comosantes principales
        col_coords = self.column_coordinates(X)
        ax.scatter(
            col_coords[x_component],
            col_coords[y_component],
            **kwargs,
            label=col_label
        )

        # Ajouter les labels des colonne
        if show_row_labels:
            x = row_coords[x_component]
            y = row_coords[y_component]
            for xi, yi, label in zip(x, y, row_names):
                ax.annotate(label, (xi, yi))
        if show_col_labels:
            x = col_coords[x_component]
            y = col_coords[y_component]
            for xi, yi, label in zip(x, y, col_names):
                ax.annotate(label, (xi, yi))

        # disposition de la legende
        ax.legend()

        # titre des plots
        ax.set_title('Principal coordinates')
        ei = self.explained_inertia_
        ax.set_xlabel('composante inertie 1 {} ({:.2f}%)'.format(x_component, 100 * ei[x_component]))
        ax.set_ylabel('composante inertie 2 {} ({:.2f}%)'.format(y_component, 100 * ei[y_component]))

        return ax
