.. _mars:

*************************
Mars 使用指南
*************************


Mars 简介
---------

Mars 能利用并行和分布式技术，加速 Python 数据科学栈，包括 `numpy <https://numpy.org/>`__\ 、\ `pandas <https://pandas.pydata.org/>`__ 和 `scikit-learn <https://scikit-learn.org/>`__\ 。同时，也能轻松与 TensorFlow、PyTorch 和 XGBoost 集成。

`Mars tensor <https://docs.pymars.org/zh_CN/latest/tensor/index.html>`__ 的接口和 numpy 保持一致，但支持大规模高维数组。样例代码如下。

.. code:: python

    import mars.tensor as mt

    a = mt.random.rand(10000, 50)
    b = mt.random.rand(50, 5000)
    a.dot(b).execute()

`Mars DataFrame <https://docs.pymars.org/zh_CN/latest/dataframe/index.html>`__ 接口和 pandas 保持一致，但可以支撑大规模数据处理和分析。样例代码如下。

.. code:: python

    import mars.dataframe as md

    ratings = md.read_csv('Downloads/ml-20m/ratings.csv')
    movies = md.read_csv('Downloads/ml-20m/movies.csv')
    movie_rating = ratings.groupby('movieId', as_index=False).agg({'rating': 'mean'})
    result = movie_rating.merge(movies[['movieId', 'title']], on='movieId')
    result.sort_values(by='rating', ascending=False).execute()

`Mars learn <https://docs.pymars.org/zh_CN/latest/learn/index.html>`__ 保持和 scikit-learn 接口一致。样例代码如下。

.. code:: python

    import mars.dataframe as md
    from mars.learn.neighbors import NearestNeighbors

    df = md.read_csv('data.csv')
    nn = NearestNeighbors(n_neighbors=10)
    nn.fit(df)
    neighbors = nn.kneighbors(df).fetch()

Mars learn 可以很方便地与 `TensorFlow <https://docs.pymars.org/zh_CN/latest/learn/tensorflow.html>`__\ 、PyTorch 和 `XGBoost <https://docs.pymars.org/zh_CN/latest/learn/xgboost.html>`__ 集成，点击链接查看文档。

在 MaxCompute 上使用 Mars，我们提供了简单易用的接口来拉起 Mars 集群，用户不需要关心安装和维护集群。同时，通过 MaxCompute 拉起的 Mars，也支持直接读写 MaxCompute 表。

申请试用
--------

公共云用户我们采用申请开通的方式，需要使用请提工单申请。

环境准备
--------

要在 MaxCompute 上运行 Mars，需要有相应的运行环境。这可以分为如下几种情况。

1. 开箱即用的环境，如 dataworks，会包含所需要的依赖。
2. 其他环境，需要自己安装相关依赖。

我们分别展开。

开箱即用的环境
~~~~~~~~~~~~~~

开箱即用的环境，如 dataworks 的 **pyodps3 节点**\ ，已经包含了 PyODPS 和 Mars。

在新建的 pyodps3 节点里运行如下命令检查版本，确保满足要求。

.. code:: python

    from odps import __version__ as odps_version
    from mars import __version__ as mars_version

    print(odps_version)
    print(mars_version)

输出的第一个为 PyODPS 版本，第二个为 Mars 版本。要求 **PyODPS 至少是 0.9.0**\ 。


其他环境
~~~~~~~~

这个环境就要求通过 pip 安装 PyODPS 和 Mars。\ **Python 版本推荐使用 3.7 版本，至少需要是 3.5 版本。**

通过如下命令安装：

.. code:: bash

    pip install -U pip  # 可选，确保 pip 版本足够新
    pip install pyarrow==0.12.1  # 目前 pyarrow 版本固定到 0.12.1
    pip install git+http://gitlab-ci-token:c60faf31b1f475342c790cca880e06@gitlab-sc.alibaba-inc.com/odps/pyodps.git
    pip install pymars>=0.4.0rc1  # mars 版本需要至少是 0.4.0rc1

准备 ODPS 入口
~~~~~~~~~~~~~~

ODPS 入口是 MaxCompute 所有操作的基础：

-  对于开箱即用的环境，如 dataworks，我们会自动创建 ``o`` 即 ODPS 入口实例，因此可以不需要创建。
-  对于其他环境，需要通过 ``access_id``\ 、\ ``access_key`` 等参数创建，详细参考 :ref:`快速开始 <quick_start>` 。

基本概念
--------

-  :ref:`MaxCompute 任务实例 <instances>` ：MaxCompute 上任务以 instance 概念存在。Mars 集群也是通过一个 MaxCompute Instance 拉起。
-  :ref:`Logview 地址 <logview>` ：每个 MaxCompute instance 包含一个 logview 地址来查看任务状态。拉起 Mars 集群的 instance 也不例外。
-  Mars UI: Mars 集群拉起后，会包含一个 Web UI，通过这个 Web UI，可以查看 Mars 集群、任务状态，可以提交任务。当集群拉起后，一般来说就不需要和 MaxCompute 任务实例交互了。
-  Mars session：Mars session 和具体的执行有关，一般情况下用户不需要关心 session，因为会包含默认的 session。通过 ``o.create_mars_cluster`` 创建了 Mars 集群后，会创建默认连接到 Mars 集群的 session。
-  `Jupyter Notebook <https://jupyter.org/>`__\ ：Jupyter Notebook 是一个基于网页的用于交互式计算的应用程序，可以用来开发、文档编写、运行代码和展示结果。

基础用法
--------

创建 Mars 集群
~~~~~~~~~~~~~~

准备好环境后，接着我们就可以拉起 Mars 集群了。

有了 ``o`` 这个对象后，拉起 Mars 集群非常简单，只需要运行如下代码。

.. code:: python

    from odps import options
    options.verbose = True  # 在 dataworks pyodps3 里已经设置，所以不需要前两行代码
    client = o.create_mars_cluster(5, 4, 16, min_worker_num=3)

这个例子里指定了 worker 数量为 5 的集群，每个 worker 是4核、16G 内存的配置，\ ``min_worker_num`` 指当 worker 已经起了3个后，就可以返回 ``client`` 对象了，而不用等全部 5 个 worker 都启动再返回。Mars 集群的创建过程可能比较慢，需要耐心等待。

**注意：申请的单个 worker 内存需大于 1G，CPU 核数和内存的最佳比例为 1：4，例如单 worker 4核、16G。同时，新建的 worker 个数也不要超过 30 个，否则会对镜像服务器造成压力，如果需要使用超过 30 个 worker，请联系我们。**

这个过程中会打印 MaxCompute instance 的 logview、 Mars UI 以及 Notebook 地址。Mars UI 可以用来连接 Mars 集群，亦可以用来查看集群、任务状态。

Mars 集群的创建就是一个 MaxCompute 任务，因此也有 instance id、logview 等 MaxCompute 通用的概念。

提交作业
~~~~~~~~

Mars 集群创建的时候会设置默认 session，通过 ``.execute()`` 执行时任务会被自动提交到集群。

.. code:: python

    import mars.dataframe as md
    import mars.tensor as mt

    md.DataFrame(mt.random.rand(10, 3)).execute()  # execute 自动提交任务到创建的集群

停止并释放集群
~~~~~~~~~~~~~~

**目前一个 Mars 集群超过3天就会被自动释放**\ 。当 Mars 集群不再需要使用时，也可以通过调用 ``client.stop_server()`` 手动释放：

.. code:: python

    client.stop_server()

MaxCompute 表读写支持
~~~~~~~~~~~~~~~~~~~~~

创建了 Mars 集群后，集群内的 Mars 任务可以直读和直写 MaxCompute 表。

读表
^^^^

通过 ``o.to_mars_dataframe`` 来读取 MaxCompute 表，并返回 `Mars DataFrame <https://docs.pymars.org/zh_CN/latest/dataframe/index.html>`__\ 。

.. code:: ipython

    In [1]: df = o.to_mars_dataframe('test_mars')
    In [2]: df.head(6).execute()
    Out[2]:
           col1  col2
    0        0    0
    1        0    1
    2        0    2
    3        1    0
    4        1    1
    5        1    2

写表
^^^^

通过 ``o.persist_mars_dataframe(df, 'table_name')`` 将 Mars DataFrame 保存成 MaxCompute 表。

.. code:: ipython

    In [3]: df = o.to_mars_dataframe('test_mars')
    In [4]: df2 = df + 1
    In [5]: o.persist_mars_dataframe(df2, 'test_mars_persist')  # 保存 Mars DataFrame
    In [6]: o.get_table('test_mars_persist').to_df().head(6)  # 通过 PyODPS DataFrame 查看数据
           col1  col2
    0        1    1
    1        1    2
    2        1    3
    3        2    1
    4        2    2
    5        2    3

使用 Mars 集群自带的 Jupyter Notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

创建 Mars 集群会自动创建一个 Jupyter Notebook 以编写代码。

新建一个 Notebook 会自动设置 session，提交任务到集群。因此在这个 notebook 内也不需要显示创建 ``session``\ 。

.. code:: python

    import mars.dataframe as md

    md.DataFrame(mt.random.rand(10, 3)).sum().execute() # 在 notebook 里运行，execute 自动提交到当前集群

有一点要注意：\ **这个 notebook 不会保存你的 notebook 文件，所以要记得自行保存**\ 。

用户也可以使用自己的 notebook 连接到集群，此时参考 `使用已经创建的 Mars 集群 <#使用已经创建的-Mars-集群>`__ 。

其他用法
--------

使用已经创建的 Mars 集群
~~~~~~~~~~~~~~~~~~~~~~~~

首先，我们可以通过 instance id 重建 Mars 集群的 client。

.. code:: python

    client = o.create_mars_cluster(instance_id=**instance-id**)

如果只是想使用 Mars，可以使用 Mars session 来连接。给定 Mars UI 的地址。则：

.. code:: python

    from mars.session import new_session
    new_session('**Mars UI address**').as_default() # 设置为默认 session

获取 Mars UI 地址
~~~~~~~~~~~~~~~~~

Mars 集群创建的时候指定了 ``options.verbose=True`` 会打印 Mars UI 地址。

也可以通过 ``client.endpoint`` 来获取 Mars UI。

.. code:: python

    print(client.endpoint)

获取 Logview 地址
~~~~~~~~~~~~~~~~~

创建集群的时候指定了 ``options.verbose=True`` 会自动打印 logview。

也可以通过 ``client.get_logview_address()`` 获取 logview 地址。

.. code:: python

    print(client.get_logview_address())

获取 Jupyter Notebook 地址
~~~~~~~~~~~~~~~~~~~~~~~~~~

Mars 集群创建的时候指定了 ``options.verbose=True`` 会打印 Jupyter Notebook 地址。

也可以通过 ``client.get_notebook_endpoint()`` 获取 Jupyter Notebook 地址。

.. code:: python

    print(client.get_notebook_endpoint())

Mars 和 PyODPS DataFrame 对比
-----------------------------

有同学会问，Mars 和 PyODPS DataFrame 有什么区别呢？

API
~~~

Mars DataFrame 的接口完全兼容 pandas。除了 DataFrame，Mars tensor 兼容 numpy，Mars learn 兼容 scikit-learn。

而 PyODPS 只有 DataFrame 接口，和 pandas 的接口存在着很多不同。

索引
~~~~

Mars DataFrame 有 pandas 索引的概念。

.. code:: ipython

    In [1]: import mars.dataframe as md

    In [5]: import mars.tensor as mt

    In [7]: df = md.DataFrame(mt.random.rand(10, 3), index=md.date_range('2020-5-1', periods=10))

    In [9]: df.loc['2020-5'].execute()
    Out[9]:
                       0         1         2
    2020-05-01  0.061912  0.507101  0.372242
    2020-05-02  0.833663  0.818519  0.943887
    2020-05-03  0.579214  0.573056  0.319786
    2020-05-04  0.476143  0.245831  0.434038
    2020-05-05  0.444866  0.465851  0.445263
    2020-05-06  0.654311  0.972639  0.443985
    2020-05-07  0.276574  0.096421  0.264799
    2020-05-08  0.106188  0.921479  0.202131
    2020-05-09  0.281736  0.465473  0.003585
    2020-05-10  0.400000  0.451150  0.956905

PyODPS 里没有索引的概念，因此跟索引有关的操作全部都不支持。

数据顺序
~~~~~~~~

Mars DataFrame 一旦创建，保证顺序，因此一些时序操作比如 ``shift``\ ，以及向前向后填空值如\ ``ffill``\ 、\ ``bfill``\ ，只有 Mars DataFrame 支持。

.. code:: ipython

    In [3]: df = md.DataFrame([[1, None], [None, 1]])

    In [4]: df.execute()
    Out[4]:
         0    1
    0  1.0  NaN
    1  NaN  1.0

    In [5]: df.ffill().execute() # 空值用上一行的值
    Out[5]:
         0    1
    0  1.0  NaN
    1  1.0  1.0

PyODPS 由于背后使用 MaxCompute 计算和存储数据，而 MaxCompute 并不保证数据顺序，所以这些操作再 MaxCompute 上都无法支持。

执行层
~~~~~~

**PyODPS 本身只是个客户端，不包含任何服务端部分。**\ PyODPS DataFrame 在真正执行时，会将计算编译到 MaxCompute SQL 执行。因此，PyODPS DataFrame 支持的操作，取决于 MaxCompute SQL 本身。此外，每一次调用 ``execute`` 方法时，会提交一次 MaxCompute 作业，需要在集群内调度。

**Mars 本身包含客户端和分布式执行层。**\ 通过调用 ``o.create_mars_cluster`` ，会在 MaxCompute 内部拉起 Mars 集群，一旦 Mars 集群拉起，后续的交互就直接和 Mars 集群进行。计算会直接提交到这个集群，调度开销极小。在数据规模不是特别大的时候，Mars 应更有优势。

使用场景指引
------------

有同学会关心，何时使用 Mars，何时使用 PyODPS DataFrame？我们分别阐述。

适合 Mars 的使用场景。
~~~~~~~~~~~~~~~~~~~~~~

-  如果你经常使用 PyODPS DataFrame 的 ``to_pandas()`` 方法，将 PyODPS DataFrame 转成 pandas DataFrame，推荐使用 Mars DataFrame。
-  Mars DataFrame 目标是完全兼容 pandas 的接口以及行为，如果你熟悉 pandas 的接口，而不愿意学习 PyODPS DataFrame 的接口，那么使用 Mars。
-  Mars DataFrame 因为兼容 pandas 的行为，因此如下的特性如果你需要用到，那么使用 Mars。
-  Mars DataFrame 包含行和列索引，如果需要使用索引，使用 Mars。
-  Mars DataFrame 创建后会保证顺序，通过 iloc 等接口可以获取某个偏移的数据。如 ``df.iloc[10]`` 可以获取第10行数据。此外，如 ``df.shift()`` 、\ ``df.ffill()`` 等需要有保证顺序特性的接口也在 Mars DataFrame 里得到了实现，有这方面的需求可以使用 Mars。
-  Mars 还包含 `Mars tensor <https://docs.pymars.org/zh_CN/latest/tensor/index.html>`__ 来并行和分布式化 Numpy，以及 `Mars learn <https://docs.pymars.org/zh_CN/latest/learn/index.html>`__ 来并行和分布式化 scikit-learn、以及支持在 Mars 集群里分布式运行 TensorFlow、PyTorch 和 XGBoost。有这方面的需求使用 Mars。

-  Mars 集群一旦创建，后续不再需要通过 MaxCompute 调度，任务可以直接提交到 Mars 集群执行；此外，Mars 对于中小型任务（数据量 T 级别以下），会有较好的性能。这些情况可以使用 Mars。

适合 PyODPS DataFrame 的使用场景
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  PyODPS DataFrame 会把 DataFrame 任务编译成 MaxCompute SQL 执行，如果希望依托 MaxCompute 调度任务，使用 PyODPS DataFrame。
-  PyODPS DataFrame 会编译任务到 MaxCompute 执行，由于 MaxCompute 相当稳定，而 Mars 相对比较新，如果对稳定性有很高要求，那么使用 PyODPS DataFrame。
-  数据量特别大（T 级别以上），使用 PyODPS DataFrame。

Mars 参考文档
-------------

-  Mars 开源地址：https://github.com/mars-project/mars
-  Mars 文档：https://docs.pymars.org/zh\_CN/latest/
-  Mars 团队专栏：https://zhuanlan.zhihu.com/mars-project

FAQ
---

Q：一个用户创建的 Mars 集群，别人能不能用。

A：可以，参考 `使用已经创建的 Mars 集群 <#使用已经创建的-Mars-集群>`__ 。
