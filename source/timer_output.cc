#include <aspect/timer_output.h>



namespace aspect
{
  MyTimerOutput::MyTimerOutput (const MPI_Comm      mpi_comm)
    :
    mpi_communicator (mpi_comm),
    ignore_first(true)
  {}


  MyTimerOutput::MyTimerOutput (const MPI_Comm      mpi_comm,
                                const bool ignore_first)
    :
    mpi_communicator (mpi_comm),
    ignore_first(ignore_first)
  {}

  void
  MyTimerOutput::enter_subsection (const std::string &section_name, const bool continue_timer)
  {
    Assert (section_name.empty() == false,
            ExcMessage ("Section string is empty."));

    Assert (std::find (active_sections.begin(), active_sections.end(),
                       section_name) == active_sections.end(),
            ExcMessage (std::string("Cannot enter the already active section <")
                        + section_name + ">."));

    if (sections.find (section_name) == sections.end())
      {
        if (mpi_communicator != MPI_COMM_SELF)
          {
            // create a new timer for this section. the second argument
            // will ensure that we have an MPI barrier before starting
            // and stopping a timer, and this ensures that we get the
            // maximum run time for this section over all processors.
            // The mpi_communicator from TimerOutput is passed to the
            // Timer here, so this Timer will collect timing information
            // among all processes inside mpi_communicator.
            sections[section_name].timer = Timer(mpi_communicator, true);

            if (!continue_timer)
              sections[section_name].time_vec.clear();
          }

        sections[section_name].n_calls = 0;
      }

    sections[section_name].timer.restart();
    sections[section_name].n_calls++;

    active_sections.push_back (section_name);
  }

  void
  MyTimerOutput::leave_subsection (const std::string &section_name, const bool continue_timer)
  {
    Assert (!active_sections.empty(),
            ExcMessage("Cannot exit any section because none has been entered!"));

    if (section_name != "")
      {
        Assert (sections.find (section_name) != sections.end(),
                ExcMessage ("Cannot delete a section that was never created."));
        Assert (std::find (active_sections.begin(), active_sections.end(),
                           section_name) != active_sections.end(),
                ExcMessage ("Cannot delete a section that has not been entered."));
      }

    // if no string is given, exit the last
    // active section.
    const std::string actual_section_name = (section_name == "" ?
                                             active_sections.back () :
                                             section_name);
    sections[actual_section_name].timer.stop();


    if (!ignore_first)
      {
        if (!continue_timer)
          sections[actual_section_name].time_vec.push_back(sections[actual_section_name].timer.last_wall_time());
        else
          sections[actual_section_name].time_vec[sections[actual_section_name].n_calls-1] += sections[actual_section_name].timer.last_wall_time();
      }
    else if (sections[actual_section_name].n_calls != 1)
      {
        if (!continue_timer)
          sections[actual_section_name].time_vec.push_back(sections[actual_section_name].timer.last_wall_time());
        else
          sections[actual_section_name].time_vec[sections[actual_section_name].n_calls-2] += sections[actual_section_name].timer.last_wall_time();
      }


    // delete the index from the list of
    // active ones
    active_sections.erase (std::find (active_sections.begin(), active_sections.end(),
                                      actual_section_name));
  }

  std::map<std::string, std::vector<double> >
  MyTimerOutput::get_all_times() const
  {
    std::map<std::string, std::vector<double> > output;
    for (const auto &section : sections)
      output[section.first] = section.second.time_vec;

    return output;
  }

  void
  MyTimerOutput::print_data_file(const std::string &filename_and_path) const
  {
    std::ofstream out;
    out.open(filename_and_path);

    out.close();
  }




  void
  MyTimerOutput::reset ()
  {
    sections.clear();
    active_sections.clear();
    timer.restart();
  }
}
