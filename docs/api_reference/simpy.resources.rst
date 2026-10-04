==================================================
``simpy.resources`` --- Shared resource primitives
==================================================

.. automodule:: simpy.resources


Resources --- ``simpy.resources.resource``
==========================================

.. automodule:: simpy.resources.resource

.. autoclass:: Resource
   :members:

.. autoclass:: PriorityResource
   :members:

.. autoclass:: PreemptiveResource
   :members:

.. autoclass:: Preempted
   :members:

.. autoclass:: Request
   :members:

.. autoclass:: PriorityRequest
   :members:

.. autoclass:: Release
   :members:

.. autoclass:: SortedQueue
   :members:


Capacity calendars --- ``simpy.resources.calendar``
===================================================

.. automodule:: simpy.resources.calendar

.. autoclass:: CalendarResource
   :members:

.. autoclass:: CapacityCalendar
   :members:

.. autoclass:: CalendarEntry
   :members:

.. autoclass:: CalendarPreemption
   :members:


Containers --- ``simpy.resources.container``
============================================

.. automodule:: simpy.resources.container

.. autoclass:: Container
   :members:

.. autoclass:: ContainerPut
   :members:

.. autoclass:: ContainerGet
   :members:


Stores --- ``simpy.resources.store``
====================================

.. automodule:: simpy.resources.store

.. autoclass:: Store
   :members:

.. autoclass:: PriorityItem(priority, item)
   :members: priority, item

.. autoclass:: PriorityStore
   :members:

.. autoclass:: FilterStore
   :members:

.. autoclass:: StorePut
   :members:

.. autoclass:: StoreGet
   :members:

.. autoclass:: FilterStoreGet
   :members:


Base classes --- ``simpy.resources.base``
=========================================

.. automodule:: simpy.resources.base

.. autoclass:: BaseResource
   :members:

.. autoclass:: Put
   :members:

.. autoclass:: Get
   :members:
