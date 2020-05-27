"""
The CobwebPlus module contains the :class:`CobwebPlusTree` and :class:`CobwebPlusNode`
classes, which extend the traditional Cobweb capabilities to be able to cluster
textual data following the paper of Sahoo et al.: Incremental Hierarchical Clustering of Text Documents
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from math import log

from concept_formation.cobweb import CobwebNode
from concept_formation.cobweb import CobwebTree
#from concept_formation.continuous_value import ContinuousValue
from concept_formation.utils import isInteger
#from concept_formation.utils import weighted_choice
#from concept_formation.utils import most_likely_choice

#cv_key = "#ContinuousValue#"


class CobwebPlusTree(CobwebTree):
    """
    The CobwebPlusTree contains the knowledge base of a particuluar instance of
    the Cobweb+ algorithm and can be used to fit and categorize instances.
    Cobweb+'s main difference over Cobweb is its application on textual data
    due to handling word counts as attribute/value pairs. Furthermore the category
    utility is adapted using Katz distribution to modelise the word occurrences.
    For the purposes of Cobweb+'s core algorithms a integer is any value
    where ``isinstance(instance, int)`` returns ``True``.

    The category utility is calculated as follows:

    .. math::

        CU_p(\\{C_1, C_2, \\cdots, C_n\\}) = \\frac{1}{n} \\sum_{k=1}^n
            P(C_k) \\left[ \\sum_i CU_{i,k} - \\sum_i CU_{i,p} \\right]

    where

    .. math::

        CU_{i,k} = \\frac{1-2p_0(1-p_0) - p(1-2p_0)}{1+p}

    with

    ..math::

        p_0 = 1-\\frac{df}{N}
        p = \\frac{cf-df}{cf}

    where
    :math:`df` = document frequency (number of documents in the entire collection that contain the word i)
    :math:`cf` = collection frequency (number of times word i occured in document collection)
    :math:`N` = number of documents in current collection
    When refering to df and cf we only consider documents from the current or child node

    # The scaling parameter determines whether online normalization of continuous
    # attributes is used, and to what standard deviation the values are scaled
    # to. Scaling divides the std of each attribute by the std of the attribute
    # in the root divided by the scaling constant (i.e., :math:`\\sigma_{root} /
    # scaling` when making category utility calculations.  Scaling is useful to
    # balance the weight of different numerical attributes, without scaling the
    # magnitude of numerical attributes can affect category utility calculation
    # meaning numbers that are naturally larger will recieve preference in the
    # category utility calculation.
    #
    # :param scaling: The number of standard deviations numeric attributes
    #     are scaled to. By default this value is 0.5 (half a standard
    #     deviation), which is the max std of nominal values. If disabiling
    #     scaling is desirable, then it can be set to False or None.
    # :type scaling: a float greater than 0.0, None, or False
    # :param inner_attr_scaling: Whether to use the inner most attribute name
    #     when scaling numeric attributes. For example, if `('attr', '?o1')` was
    #     an attribute, then the inner most attribute would be 'attr'. When using
    #     inner most attributes, some objects might have multiple attributes
    #     (i.e., 'attr' for different objects) that contribute to the scaling.
    # :param inner_attr_scaling: boolean
    """

    def __init__(self):
        """
        The tree constructor.
        """
        self.root = CobwebPlusNode()
        self.root.tree = self
        #self.attr_scales = {}

    def clear(self):
        """
        Clears the concepts of the tree, but maintains the scaling parameter.
        """
        self.root = CobwebPlusNode()
        self.root.tree = self
        #self.attr_scales = {}

#     def get_inner_attr(self, attr):
#         """
#         Extracts the inner most attribute name from the provided attribute, if
#         the attribute is a tuple and inner_attr_scaling is on. Otherwise it
#         just returns the attribute. This is used to for normalizing attributes.
#
#         >>> t = Cobweb3Tree()
#         >>> t.get_inner_attr(('a', '?object1'))
#         'a'
#         >>> t.get_inner_attr('a')
#         'a'
#         """
#         if isinstance(attr, tuple) and self.inner_attr_scaling:
#             return attr[0]
#         else:
#             return attr
#
#     def update_scales(self, instance):
#         """
#         Reads through all the attributes in an instance and updates the
#         tree scales object so that the attributes can be properly scaled.
#         """
#         for attr in instance:
#             if isNumber(instance[attr]):
#                 inner_attr = self.get_inner_attr(attr)
#                 if inner_attr not in self.attr_scales:
#                     self.attr_scales[inner_attr] = ContinuousValue()
#                 self.attr_scales[inner_attr].update(instance[attr])
#
    def cobweb(self, instance):
        """
        A modification of the cobweb function to update the scales object
        first, so that attribute values can be properly scaled.
        """
        #self.update_scales(instance)
        return super(CobwebPlusTree, self).cobweb(instance)

    def ifit(self, instance):
        """
        Incrementally fit a new instance into the tree and return its resulting
        concept.

        The cobweb+ version of the :meth:`CobwebTree.ifit` function. This
        version keeps track of all of the continuous

        :param instance: An instance to be categorized into the tree.
        :type instance:  :ref:`Instance<instance-rep>`
        :return: A concept describing the instance
        :rtype: CobwebPlusNode

        .. seealso:: :meth:`CobwebTree.cobweb`
        """
        self._sanity_check_instance(instance)
        return self.cobweb(instance)


class CobwebPlusNode(CobwebNode):
    """
    A CobwebPlusNode represents a concept/cluster within the knoweldge base of a
    particular :class:`CobwebPlusTree`. Each node contains a probability table
    that can be used to calculate the probability of different attributes/words given
    the concept that the node represents.

    In general the :meth:`CobwebPlusTree.ifit`, :meth:`CobwebPlusTree.categorize`
    functions should be used to initially interface with the Cobweb+ knowledge
    base and then the returned concept can be used to calculate probabilities
    of certain attributes or determine concept labels.
    """

    def increment_counts(self, instance):
        """
        Increment the counts at the current node according to the specified
        instance.

        CobwebPlusNode uses a modified version of
        :meth:`CobwebNode.increment_counts
        <concept_formation.cobweb.CobwebNode.increment_counts>` that handles
        integer attributes properly. Any attribute value where
        ``isinstance(instance[attr], int)`` returns ``True`` will be treated
        as a integer attribute.

        Ex.:
        .. math ::
            avcounts_k = \\{word i : \\{"cf" : Number of times word i occured in document collection,
                                        "df" : Number of documents in entire collection that contain word i\\}\\}

        # .. warning:: If a numeric attribute is found in an instance with the
        #     name of a previously nominal attribute, or vice versa, this
        #     function will raise an exception. See: :class:`NumericToNominal
        #     <concept_formation.preprocessor.NumericToNominal>` for a way to fix
        #     this error.

        :param instance: A new instances to incorporate into the node.
        :type instance: :ref:`Instance<instance-rep>`

        """
        print("increment_counts:")
        print(instance)
        self.count += 1
        # instance:  words -> {collectionFrequency -> count, documentFrequency -> count}
        for attr in instance:
            if isInteger(instance[attr]):
                if attr not in self.av_counts:
                    self.av_counts[attr] = {"cf" : 0, "df" : 0}
                self.av_counts[attr]["cf"] += instance[attr]
                self.av_counts[attr]["df"] += 1
            else:
                raise Exception("Word counts should always be Integers!")

    def update_counts_from_node(self, node):
        """
        Increments the counts of the current node by the amount in the
        specified node, modified to handle numbers.

        This function is used as part of copying nodes and in merging nodes

        :param node: Another node from the same CobwebPlusTree
        :type node: CobwebPlusNode
        """
        #print("!!\nupdate_counts_from_node:")
        self.count += node.count
        for attr in node.attrs('all'):
            if attr not in self.av_counts:
                self.av_counts[attr] = {"cf" : 0, "df" : 0}
            self.av_counts[attr]["cf"] += node.av_counts[attr]["cf"]
            self.av_counts[attr]["df"] += node.av_counts[attr]["df"]


    def expected_correct_guesses(self):
        """
        Returns the number of attribute values that would be correctly guessed
        in the current concept.

        The typical Cobweb+ calculation for correct guesses is:

        .. math::

            P(A_i = V_{ij})^2 = \\frac{1}{n} \\sum_{k=1}^n
                P(C_k) \\left[ \\sum_i CU_{i,k} - \\sum_i CU_{i,p} \\right]

        where

        .. math::

            CU_{i,k} = \\frac{1-2p_0(1-p_0) - p(1-2p_0)}{1+p}

        with

        ..math::

            p_0 = 1-\\frac{df}{N}
            p = \\frac{cf-df}{cf}

        where
        :math:`df` = document frequency (number of documents in the entire collection that contain the word i)
        :math:`cf` = collection frequency (number of times word i occured in document collection)
        :math:`N` = number of documents in current collection
        When refering to df and cf we only consider documents from the current or child node

        :return: The number of attribute values that would be correctly guessed
            in the current concept.
        :rtype: float
        """
        correct_guesses = 0.0
        # TODO check if attr_count needed
        attr_count = 0
        for attr in self.attrs():
            attr_count += 1
            p0 = 1 - self.av_counts[attr]["df"] / self.count
            p = (self.av_counts[attr]["cf"] - self.av_counts[attr]["df"]) / self.av_counts[attr]["cf"]
            #print("attr: {}, count: {}, df: {}, cf: {}".format(attr, self.count, self.av_counts[attr]["df"], self.av_counts[attr]["cf"]))
            correct_guesses += ( (1-2*p0*(1-p0) - p*(1-2*p0)) / (1+p) )
            #print("p0: {}, p: {}, correct_guesses: {}\n".format(p0, p, correct_guesses))
        return correct_guesses #/ attr_count


    def is_exact_match(self, instance):
        """
        Returns true if the concept exactly matches the instance.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: whether the instance perfectly matches the concept
        :rtype: boolean

        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        for attr in set(instance).union(set(self.attrs())):
            if attr[0] == '_':
                continue
            if attr in instance and attr not in self.av_counts:
                return False
            if attr in self.av_counts and attr not in instance:
                return False

        return True


#     def pretty_print(self, depth=0):
#         """
#         Print the categorization tree
#
#         The string formatting inserts tab characters to align child nodes of
#         the same depth. Numerical values are printed with their means and
#         standard deviations.
#
#         :param depth: The current depth in the print, intended to be called
#             recursively
#         :type depth: int
#         :return: a formated string displaying the tree and its children
#         :rtype: str
#         """
#         ret = str(('\t' * depth) + "|-")
#
#         attributes = []
#
#         for attr in self.attrs('all'):
#             values = []
#             for val in self.av_counts[attr]:
#                 values.append("'" + str(val) + "': " +
#                               str(self.av_counts[attr][val]))
#
#             attributes.append("'" + str(attr) + "': {" + ", ".join(values)
#                               + "}")
#
#         ret += "{" + ", ".join(attributes) + "}: " + str(self.count) + '\n'
#
#         for c in self.children:
#             ret += c.pretty_print(depth+1)
#
#         return ret
#
#     def get_weighted_values(self, attr, allow_none=True):
#         """
#         Return a list of weighted choices for an attribute based on the node's
#         probability table.
#
#         This calculation will include an option for the change that an
#         attribute is missing from an instance all together. This is useful for
#         probability and sampling calculations. If the attribute has never
#         appeared in the tree then it will return a 100% chance of None.
#
#         :param attr: an attribute of an instance
#         :type attr: :ref:`Attribute<attributes>`
#         :param allow_none: whether attributes in the nodes probability table
#             can be inferred to be missing. If False, then None will not be
#             cosidered as a possible value.
#         :type allow_none: Boolean
#         :return: a list of weighted choices for attr's value
#         :rtype: [(:ref:`Value<values>`, float), (:ref:`Value<values>`, float),
#             ...]
#         """
#         choices = []
#         if attr not in self.av_counts:
#             choices.append((None, 1.0))
#             return choices
#
#         val_count = 0
#         for val in self.av_counts[attr]:
#             if val == cv_key:
#                 count = self.av_counts[attr][val].num
#             else:
#                 count = self.av_counts[attr][val]
#             choices.append((val, count / self.count))
#             val_count += count
#
#         if allow_none:
#             choices.append((None, ((self.count - val_count) / self.count)))
#
#         return choices
#
#     def predict(self, attr, choice_fn="most likely", allow_none=True):
#         """
#         Predict the value of an attribute, using the provided strategy.
#
#         If the attribute is a nominal then this function behaves the same as
#         :meth:`CobwebNode.predict
#         <concept_formation.cobweb.CobwebNode.predict>`.  If the attribute is
#         numeric then the mean value from the
#         :class:`ContinuousValue<concept_formation.cv_key.ContinuousValue>` is
#         chosen.
#
#         :param attr: an attribute of an instance.
#         :type attr: :ref:`Attribute<attributes>`
#         :param allow_none: whether attributes not in the instance can be
#             inferred to be missing. If False, then all attributes will be
#             inferred with some value.
#         :type allow_none: Boolean
#         :return: The most likely value for the given attribute in the node's
#             probability table.
#         :rtype: :ref:`Value<values>`
#
#         .. seealso :meth:`Cobweb3Node.sample`
#         """
#         if choice_fn == "most likely" or choice_fn == "m":
#             choose = most_likely_choice
#         elif choice_fn == "sampled" or choice_fn == "s":
#             choose = weighted_choice
#         else:
#             raise Exception("Unknown choice_fn")
#
#         if attr not in self.av_counts:
#             return None
#
#         choices = self.get_weighted_values(attr, allow_none)
#         val = choose(choices)
#
#         if val == cv_key:
#             if choice_fn == "most likely" or choice_fn == "m":
#                 val = self.av_counts[attr][val].mean
#             elif choice_fn == "sampled" or choice_fn == "s":
#                 val = normalvariate(self.av_counts[attr][val].unbiased_mean(),
#                                     self.av_counts[attr][val].unbiased_std())
#             else:
#                 raise Exception("Unknown choice_fn")
#
#         return val
#
#     def probability(self, attr, val):
#         """
#         Returns the probability of a particular attribute value at the current
#         concept.
#
#         This takes into account the possibilities that an attribute can take
#         any of the values available at the root, or be missing.
#
#         For numerical attributes it returns the integral of the product of two
#         gaussians. One gaussian has :math:`\\mu = val` and :math:`\\sigma =
#         \\sigma_{noise} = \\frac{1}{2 * \\sqrt{\\pi}}` (where
#         :math:`\\sigma_{noise}` is from
#         :meth:`Cobweb3Node.expected_correct_guesses
#         <concept_formation.cobweb3.Cobweb3Node.expected_correct_guesses>` and
#         ensures the probability or expected correct guesses never exceeds 1).
#         The second gaussian has the mean ad std values from the current concept
#         with additional gaussian noise (independent and normally distributed
#         noise with :math:`\\sigma_{noise} = \\frac{1}{2 * \\sqrt{\\pi}}`).
#
#         The integral of this gaussian product is another gaussian with
#         :math:`\\mu` equal to the concept attribut mean and :math:`\\sigma =
#         \\sqrt{\\sigma_{attr}^2 + 2 * \\sigma_{noise}^2}` or, slightly
#         simplified, :math:`\\sigma =
#         \\sqrt{\\sigma_{attr}^2 + 2 * \\frac{1}{2 * \\pi}}`.
#
#         :param attr: an attribute of an instance
#         :type attr: :ref:`Attribute<attributes>`
#         :param val: a value for the given attribute
#         :type val: :ref:`Value<values>`
#         :return: The probability of attr having the value val in the current
#             concept.
#         :rtype: float
#         """
#         if val is None:
#             c = 0.0
#             if attr in self.av_counts:
#                 c = sum([self.av_counts[attr][v].num if v == cv_key
#                          else self.av_counts[attr][v] for v in
#                          self.av_counts[attr]])
#             return (self.count - c) / self.count
#
#         if attr in self.av_counts and isNumber(val):
#             if cv_key not in self.av_counts[attr]:
#                 return 0.0
#
#             prob_attr = self.av_counts[attr][cv_key].num / self.count
#             if self.tree is not None and self.tree.scaling:
#                 inner_attr = self.tree.get_inner_attr(attr)
#                 scale = ((1/self.tree.scaling) *
#                          self.tree.attr_scales[inner_attr].unbiased_std())
#
#                 if scale == 0:
#                     scale = 1
#                 shift = self.tree.attr_scales[inner_attr].mean
#                 val = (val - shift) / scale
#             else:
#                 scale = 1.0
#                 shift = 0.0
#
#             mean = (self.av_counts[attr][cv_key].mean - shift) / scale
#             ostd = self.av_counts[attr][cv_key].scaled_unbiased_std(scale)
#             std = sqrt(ostd * ostd + (1 / (2 * pi)))
#             p = (prob_attr *
#                  (1/(sqrt(2*pi) * std)) *
#                  exp(-((val - mean) * (val - mean)) / (2.0 * std * std)))
#             return p
#
#         if attr in self.av_counts and val in self.av_counts[attr]:
#             return self.av_counts[attr][val] / self.count
#
#         return 0.0
#
#     def log_likelihood(self, child_leaf):
#         """
#         Returns the log-likelihood of a leaf contained within the current
#         concept. Note, if the leaf contains multiple instances, then it is
#         treated as if it contained just a single instance (this function is
#         just called multiple times for each instance in the leaf).
#         """
#         ll = 0
#         for attr in set(self.attrs()).union(set(child_leaf.attrs())):
#             vals = set([None])
#             if attr in self.av_counts:
#                 vals.update(self.av_counts[attr])
#             if attr in child_leaf.av_counts:
#                 vals.update(child_leaf.av_counts[attr])
#
#             for val in vals:
#                 if val == cv_key:
#                     if (attr in self.av_counts and cv_key in
#                             self.av_counts[attr] and attr in
#                             child_leaf.av_counts and cv_key in
#                             child_leaf.av_counts[attr]):
#
#                         n1 = self.av_counts[attr][cv_key]
#                         n2 = child_leaf.av_counts[attr][cv_key]
#                         pn1 = n1.num / self.count
#                         pn2 = n2.num / child_leaf.count
#                         p = pn1 * pn2 * n1.integral_of_gaussian_product(n2)
#                         if p > 0:
#                             ll += log(p)
#                         else:
#                             raise Exception("p should be greater than 0")
#                 else:
#                     op = child_leaf.probability(attr, val)
#                     if op > 0:
#                         p = self.probability(attr, val) * op
#                         if p > 0:
#                             ll += log(p)
#                         else:
#                             raise Exception("p must be greater than 0")
#         return ll
#

#
#     def output_json(self):
#         """
#         Outputs the categorization tree in JSON form.
#
#         This is a modification of the :meth:`CobwebNode.output_json
#         <concept_formation.cobweb.CobwebNode.output_json>` to handle numeric
#         values.
#
#         :return: an object that contains all of the structural information of
#             the node and its children
#         :rtype: obj
#         """
#         output = {}
#         if "_guid" in self.av_counts:
#             for guid in self.av_counts['_guid']:
#                 output['guid'] = guid
#         output["name"] = "Concept" + str(self.concept_id)
#         output["size"] = self.count
#         output["children"] = []
#
#         temp = {}
#         for attr in self.attrs('all'):
#             temp[str(attr)] = {}
#
#             for val in self.av_counts[attr]:
#                 if val == cv_key:
#                     json_str = self.av_counts[attr][val].output_json()
#                     temp[str(attr)][cv_key] = json_str
#                 else:
#                     temp[str(attr)][str(val)] = self.av_counts[attr][val]
#
#         for child in self.children:
#             output["children"].append(child.output_json())
#
#         output["counts"] = temp
#
#         return output
