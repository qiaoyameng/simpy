================================================
``simpy.phased`` --- Staged (phased) environment
================================================

.. automodule:: simpy.phased

.. autoclass:: Phase
    :members:

.. autoclass:: PhasedEnvironment

    .. autoattribute:: now
    .. autoattribute:: microstep
    .. autoattribute:: phase
    .. autoattribute:: microstep_limit
    .. autoattribute:: active_process

    .. automethod:: schedule
    .. automethod:: peek
    .. automethod:: step
    .. automethod:: run

.. autoclass:: MicrostepLimit
    :members:
