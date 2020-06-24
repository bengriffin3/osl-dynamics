"""Helper functions using NumPy

"""
import logging
from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from vrad.utils.decorators import transpose


@transpose
def correlate_states(
    state_time_course_1: np.ndarray, state_time_course_2: np.ndarray
) -> np.ndarray:
    """Calculate the correlation matrix between states in two state-time-courses.

    Given two state time courses, calculate the correlation between each pair of states
    in the state time courses. The output for each value in the matrix is the value
    numpy.corrcoef(state_time_course_1, state_time_course_2)[0, 1].

    Parameters
    ----------
    state_time_course_1: numpy.ndarray
    state_time_course_2: numpy.ndarray

    Returns
    -------
    correlation_matrix: numpy.ndarray
    """
    correlation = np.zeros((state_time_course_1.shape[1], state_time_course_2.shape[1]))
    for i, state1 in enumerate(state_time_course_1.T):
        for j, state2 in enumerate(state_time_course_2.T):
            correlation[i, j] = np.corrcoef(state1, state2)[0, 1]
    return correlation


@transpose
def match_states(*state_time_courses: np.ndarray) -> List[np.ndarray]:
    """Find correlated states between state time courses.

    Given N state time courses and using the first given state time course as a basis,
    find the best matches for states between all of the state time courses. Once found,
    the state time courses are returned with the states reordered so that the states
    match.

    Given two arrays with columns ABCD and CBAD, both will be returned with states in
    the order ABCD.

    Parameters
    ----------
    state_time_courses: list of numpy.ndarray

    Returns
    -------
    matched_state_time_courses: list of numpy.ndarray
    """
    matched_state_time_courses = [state_time_courses[0]]
    for state_time_course in state_time_courses[1:]:

        correlation = correlate_states(state_time_courses[0], state_time_course)
        correlation = np.nan_to_num(correlation, nan=np.nanmin(correlation) - 1)
        matches = linear_sum_assignment(-correlation)
        matched_state_time_courses.append(state_time_course[:, matches[1]])
    return matched_state_time_courses


def get_one_hot(values: np.ndarray, n_states: int = None):
    """Expand a categorical variable to a series of boolean columns (one-hot encoding).

    +----------------------+
    | Categorical Variable |
    +======================+
    |           A          |
    +----------------------+
    |           C          |
    +----------------------+
    |           D          |
    +----------------------+
    |           B          |
    +----------------------+

    becomes

    +---+---+---+---+
    | A | B | C | D |
    +===+===+===+===+
    | 1 | 0 | 0 | 0 |
    +---+---+---+---+
    | 0 | 0 | 1 | 0 |
    +---+---+---+---+
    | 0 | 0 | 0 | 1 |
    +---+---+---+---+
    | 0 | 1 | 0 | 0 |
    +---+---+---+---+

    Parameters
    ----------
    values : numpy.ndarray
        Categorical variable in a 1D array. Values should be integers (i.e. state 0, 1,
        2, 3, ... , `n_states`).
    n_states : int
        Total number of states in `values`. Must be at least the number of states
        present in `values`. Default is the number of unique values in `values`.

    Returns
    -------
    one_hot : numpy.ndarray
        A 2D array containing the one-hot encoded form of the input data.

    """
    if values.ndim == 2:
        logging.info("argmax being taken on shorter axis.")
        values = values.argmax(axis=1)
    if n_states is None:
        n_states = values.max() + 1
    res = np.eye(n_states)[np.array(values).reshape(-1)]
    return res.reshape(list(values.shape) + [n_states])


@transpose(0, "sequence_1", 1, "sequence_2")
def align_arrays(*sequences, alignment: str = "left") -> List[np.ndarray]:
    """Given a list of sequences, return the sequences trimmed to equal length.

    Given a list of sequences of unequal length, remove either the start, end or a
    portion of both the start and end of the arrays such that their lengths are equal
    to the length of the shortest array.

    If alignment is "left", values will be trimmed from the ends of the arrays
    (i.e. the starts of the arrays will be aligned). If "right", values will be trimmed
    from the starts of the arrays (i.e. the ends will be aligned). If "center", an
    equal amount will be trimmed from the start and end of the array (i.e. the arrays
    are aligned by their middle values.


    Parameters
    ----------
    sequences: list of numpy.ndarray
        Time courses with differing lengths.
    alignment: str
        One of "left", "center" and "right".
    Returns
    -------
    aligned_arrays: list of numpy.ndarray
    """
    min_length = min(len(sequence) for sequence in sequences)

    if alignment == "left":
        return [sequence[:min_length] for sequence in sequences]

    elif alignment == "right":
        return [sequence[-min_length:] for sequence in sequences]
    elif alignment == "center":
        half_length = int(min_length / 2)
        mids = [int(len(sequence) / 2) for sequence in sequences]

        return [
            sequence[mid - half_length : mid + half_length]
            for sequence, mid in zip(sequences, mids)
        ]

    else:
        raise ValueError("Alignment must be left, right or center.")


@transpose(0, "state_time_course")
def state_activation(state_time_course: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate state activations for a state time course.

    Given a state time course (strictly binary), calculate the beginning and end of each
    activation of each state.

    Parameters
    ----------
    state_time_course : numpy.ndarray
        State time course (strictly binary).

    Returns
    -------
    ons : list of numpy.ndarray
        List containing state beginnings in the order they occur for each channel.
        This cannot necessarily be converted into an array as an equal number of
        elements in each array is not guaranteed.
    offs : list of numpy.ndarray
        List containing state ends in the order they occur for each channel.
        This cannot necessarily be converted into an array as an equal number of
        elements in each array is not guaranteed.

    """
    channel_on = []
    channel_off = []

    diffs = np.diff(state_time_course, axis=0)
    for i, diff in enumerate(diffs.T):
        on = (diff == 1).nonzero()[0]
        off = (diff == -1).nonzero()[0]
        try:
            if on[-1] > off[-1]:
                off = np.append(off, len(diff))

            if off[0] < on[0]:
                on = np.insert(on, 0, -1)

            channel_on.append(on)
            channel_off.append(off)
        except IndexError:
            logging.info(f"No activation in state {i}.")
            channel_on.append(np.array([]))
            channel_off.append(np.array([]))

    channel_on = np.array(channel_on)
    channel_off = np.array(channel_off)

    return channel_on, channel_off


@transpose(0, "state_time_course")
def reduce_state_time_course(state_time_course: np.ndarray) -> np.ndarray:
    """Remove empty states from a state time course.

    If a state has no activation in the state time course, remove the column
    corresponding to that state.

    Parameters
    ----------
    state_time_course: numpy.ndarray

    Returns
    -------
    reduced_state_time_course: numpy.ndarray
        A state time course with no states with no activation.

    """
    return state_time_course[:, ~np.all(state_time_course == 0, axis=0)]


@transpose(0, "state_time_course")
def state_lifetimes(state_time_course: np.ndarray) -> List[np.ndarray]:
    """Calculate state lifetimes for a state time course.

    Given a state time course (one-hot encoded), calculate the lifetime of each
    activation of each state.

    Parameters
    ----------
    state_time_course : numpy.ndarray
        State time course (strictly binary).

    Returns
    -------
    channel_lifetimes : list of numpy.ndarray
        List containing an array of lifetimes in the order they occur for each channel.
        This cannot necessarily be converted into an array as an equal number of
        elements in each array is not guaranteed.

    """
    ons, offs = state_activation(state_time_course)
    channel_lifetimes = offs - ons
    return channel_lifetimes


def from_cholesky(cholesky_matrix: np.ndarray):
    """Given a Cholesky matrix return the recomposed matrix.

    Operates on the assumption that cholesky_matrix is a valid Cholesky decomposition
    A = LL* and performs LL^T to recover A.

    Parameters
    ----------
    cholesky_matrix: numpy.ndarray
        A valid Cholesky decomposition.

    Returns
    -------
    full_matrix: numpy.ndarray
        A = LL^T where L is the Cholesky decomposition of A.
    """
    if cholesky_matrix.ndim == 2:
        return cholesky_matrix @ cholesky_matrix.transpose()
    return cholesky_matrix @ cholesky_matrix.transpose((0, 2, 1))


@transpose(0, "state_time_course")
def calculate_trans_prob_matrix(
    state_time_course: np.ndarray, zero_diagonal: bool = False, n_states: int = None,
) -> np.ndarray:
    """For a given state time course, calculate the transition probability matrix.

    If a 2D array is given, argmax(axis=1) will be performed upon it before proceeding.

    Parameters
    ----------
    state_time_course: numpy.ndarray
    zero_diagonal: bool
        If True, return the array with diagonals set to zero.
    n_states: int
        The number of states in the state time course. Default is to take the highest
        state number present in a 1D time course or the number of columns in a 2D
        (one-hot encoded) time course.

    Returns
    -------

    """
    if state_time_course.ndim == 2:
        n_states = state_time_course.shape[1]
        state_time_course = state_time_course.argmax(axis=1)
    if state_time_course.ndim != 1:
        raise ValueError("state_time_course should either be 1D or 2D.")

    vals, counts = np.unique(
        state_time_course[
            np.arange(2)[None, :] + np.arange(len(state_time_course) - 1)[:, None]
        ],
        axis=0,
        return_counts=True,
    )

    if n_states is None:
        n_states = state_time_course.max() + 1

    trans_prob = np.zeros((n_states, n_states))
    trans_prob[vals[:, 0], vals[:, 1]] = counts

    with np.errstate(divide="ignore", invalid="ignore"):
        trans_prob = trans_prob / trans_prob.sum(axis=1)[:, None]
    trans_prob = np.nan_to_num(trans_prob)

    if zero_diagonal:
        np.fill_diagonal(trans_prob, 0)
    return trans_prob


def trace_normalize(matrix: np.ndarray):
    """Given a matrix, divide all of its values by the sum of its diagonal.

    Parameters
    ----------
    matrix: numpy.ndarray

    Returns
    -------
    normalized_matrix: numpy.ndarray
        trace(M) = 1

    """
    matrix = np.array(matrix)
    if matrix.ndim == 2:
        return matrix / matrix.trace()
    if matrix.ndim != 3:
        raise ValueError("Matrix should be 2D or 3D.")

    return matrix / matrix.trace(axis1=1, axis2=2)[:, None, None]


def mean_diagonal(array: np.ndarray):
    """Set the diagonal of a matrix to the mean of all non-diagonal elements.

    This is primarily useful for plotting without being concerned about the magnitude
    of diagonal values compressing the color scale.

    Parameters
    ----------
    array: numpy.ndarray

    Returns
    -------
    mean_diagonal_array: numpy.ndarray
        Array with diagonal set to mean of non-diagonal elements.

    """
    off_diagonals = ~np.eye(array.shape[0], dtype=bool)
    new_array = array.copy()
    np.fill_diagonal(new_array, array[off_diagonals].mean())
    return new_array