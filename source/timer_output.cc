#include <aspect/timer_output.h>



namespace aspect
{
  MyTimerOutput::MyTimerOutput (const MPI_Comm      mpi_comm,
                                const unsigned int  n_timings)
    :
    mpi_communicator (mpi_comm),
    n_timings(n_timings),
    ignore_first(true)
  {}


  MyTimerOutput::MyTimerOutput (const MPI_Comm      mpi_comm,
                                const unsigned int  n_timings,
                                const bool ignore_first)
    :
    mpi_communicator (mpi_comm),
    n_timings(n_timings),
    ignore_first(ignore_first)
  {}

  void MyTimerOutput::initialize_sections()
  {
    sections.clear();
    active_sections.clear();

    std::vector<std::string> possible_sections {"total_setup", "total_assembly", "gmres_solve", "preconditioner_vmult",
                                                "setup_sys_dofs", "setup_mf_dofs", "setup_mg_dofs", "setup_mf_ops", "setup_mg_transfer",
                                                "setup_sparsity", "assemble_sys_mat_rhs", "assemble_mf_coef_rhs", "assemble_prec_mat",
                                                "assemble_amg"
                                               };

    for (const auto section_name : possible_sections)
      {
        sections[section_name].timer = Timer(mpi_communicator, true);

        sections[section_name].time_vec.clear();
        sections[section_name].time_vec.resize(n_timings,0.0);

        sections[section_name].n_calls = 0;
      }

  }

  void
  MyTimerOutput::enter_subsection (const std::string &section_name)
  {
    Assert (section_name.empty() == false,
            ExcMessage ("Section string is empty."));

    Assert (std::find (active_sections.begin(), active_sections.end(),
                       section_name) == active_sections.end(),
            ExcMessage (std::string("Cannot enter the already active section <")
                        + section_name + ">."));

    if (sections.find (section_name) == sections.end())
      {
        Assert (false, ExcMessage("All sections should be loaded in the initialize function"));

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

            sections[section_name].time_vec.clear();
            sections[section_name].time_vec.resize(n_timings,0.0);
          }

        sections[section_name].n_calls = 0;
      }

    sections[section_name].timer.restart();
    sections[section_name].n_calls++;

    active_sections.push_back (section_name);
  }

  void
  MyTimerOutput::leave_subsection (const std::string &section_name)
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

    if (!(ignore_first && sections[actual_section_name].n_calls == 1))
      {
        // if we are doing 5 timings, and we hit a 6th timing, assume it should be added to the 1st timing
        const unsigned int indx = (ignore_first ? sections[actual_section_name].n_calls-2 : sections[actual_section_name].n_calls-1)%n_timings;
        sections[actual_section_name].time_vec[indx] += sections[actual_section_name].timer.last_wall_time();
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
  MyTimerOutput::print_data_file(const std::string  &filename_and_path,
                                 const std::string  &problem_type,
                                 const unsigned int cells,
                                 const unsigned int dofs,
                                 const unsigned int procs)
  {
    std::ofstream out;
    out.open(filename_and_path);


    std::vector<std::string> possible_sections {"total_setup", "total_assembly", "gmres_solve", "preconditioner_vmult",
                                                "setup_sys_dofs", "setup_mf_dofs", "setup_mg_dofs", "setup_mf_ops", "setup_mg_transfer",
                                                "setup_sparsity", "assemble_sys_mat_rhs", "assemble_mf_coef_rhs", "assemble_prec_mat",
                                                "assemble_amg"
                                               };

    out << "type: " << problem_type << " Cells: " << cells << " DoFs: " << dofs << " Procs: " << procs << " ";

    for (const auto section_name : possible_sections)
      {
        out << section_name << ": ";

        std::sort(sections[section_name].time_vec.begin(), sections[section_name].time_vec.end());
        for (unsigned int i=0; i<n_timings; ++i)
          out << sections[section_name].time_vec[i] << " ";
      }



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
