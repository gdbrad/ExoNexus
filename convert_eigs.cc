/*! \file
 *  \brief Main program to run all measurement codes.
 */
#include "chroma.h"
#include <array>
#include <cstddef>
#include <qdp_scalarsite_defs.h>
#include <qdp_stdio.h>
#include <qdp_word.h>
#include <string>
#include "util/ferm/superb_contractions.h"
#include "qdp_db_imp.h"
const size_t num_vecs = 128;

using namespace Chroma;
extern "C" {
  void _mcleanup();
}

/*
 * Input 
 */
struct Params_t {
  multi1d<int> nrow;
  std::string inline_measurement_xml;
};

struct Inline_input_t {
  Params_t param;
  GroupXML_t cfg;
  QDP::Seed rng_seed;
};

void read(XMLReader& xml, const std::string& path, Params_t& p) {
  XMLReader paramtop(xml, path);
  read(paramtop, "nrow", p.nrow);

  XMLReader measurements_xml(paramtop, "InlineMeasurements");
  std::ostringstream inline_os;
  measurements_xml.print(inline_os);
  p.inline_measurement_xml = inline_os.str();
  QDPIO::cout << "InlineMeasurements are: " << std::endl;
  QDPIO::cout << p.inline_measurement_xml << std::endl;
}

void read(XMLReader& xml, const std::string& path, Inline_input_t& p) {
  try {
    XMLReader paramtop(xml, path);
    read(paramtop, "Param", p.param);
    p.cfg = readXMLGroup(paramtop, "Cfg", "cfg_type");

    if (paramtop.count("RNG") > 0)
      read(paramtop, "RNG", p.rng_seed);
    else
      p.rng_seed = 11; // default seed
  } catch (const std::string& e) {
    QDPIO::cerr << "Error reading XML : " << e << std::endl;
    QDP_abort(1);
  }
}

bool linkageHack(void) {
  bool foo = true;
  foo &= InlineAggregateEnv::registerAll();
  foo &= GaugeInitEnv::registerAll();
  return foo;
}

/*! \defgroup chromamain Main program to run all measurement codes.
 *  \ingroup main
 */
int main(int argc, char *argv[]) {
  Chroma::initialize(&argc, &argv);

  START_CODE();

  if (argc < 2) {
    QDPIO::cerr << "Usage: " << argv[0] << " <input.sdb>" << std::endl;
    QDP_abort(1);
  }

  std::string sdb_file = argv[1];
  QDPIO::cout << "Number of vectors: " << num_vecs << std::endl;

  QDPIO::cout << "SDB file: " << sdb_file << std::endl;
  QDPIO::cout << "Linkage = " << linkageHack() << std::endl;
  std::vector<std::string> colorvec_files = {sdb_file};
  StopWatch snoop;
  snoop.reset();
  snoop.start();

  if (Chroma::getInputFileList().empty()) {
    Chroma::getInputFileList().push_back(Chroma::getXMLInputFileName());
  }

  for (int list_index = 0; list_index < Chroma::getInputFileList().size(); ++list_index) {
    Chroma::setXMLInputFileName(Chroma::getInputFileList().at(list_index));

    XMLReader xml_in;
    Inline_input_t input;
    try {
      xml_in.open(Chroma::getXMLInputFileName());
      read(xml_in, "/chroma", input);
    } catch (const std::string& e) {
      QDPIO::cerr << "CHROMA: Caught Exception reading XML: " << e << std::endl;
      QDP_abort(1);
    } catch (std::exception& e) {
      QDPIO::cerr << "CHROMA: Caught standard library exception: " << e.what() << std::endl;
      QDP_abort(1);
    } catch (...) {
      QDPIO::cerr << "CHROMA: caught generic exception reading XML" << std::endl;
      QDP_abort(1);
    }

    XMLFileWriter& xml_out = Chroma::getXMLOutputInstance();

    if (list_index == 0) {
      push(xml_out, "chroma");
    }

    write(xml_out, "Input", xml_in);

    if (list_index == 0) {
      Layout::setLattSize(input.param.nrow);
      Layout::create();
    }

    proginfo(xml_out); // Print out basic program info
    QDP::RNG::setrn(input.rng_seed);
    write(xml_out, "RNG", input.rng_seed);

    StopWatch swatch;
    swatch.reset();
    multi1d<LatticeColorMatrix> u(Nd);
    XMLReader gauge_file_xml, gauge_xml;

    QDPIO::cout << "CHROMA: Attempt to read gauge field" << std::endl;
    swatch.start();
    try {
      std::istringstream xml_c(input.cfg.xml);
      XMLReader cfgtop(xml_c);
      QDPIO::cout << "CHROMA: Gauge initialization: cfg_type = " << input.cfg.id << std::endl;

      Handle<GaugeInit> gaugeInit(TheGaugeInitFactory::Instance().createObject(
          input.cfg.id, cfgtop, input.cfg.path));
      (*gaugeInit)(gauge_file_xml, gauge_xml, u);
    } catch (std::bad_cast) {
      QDPIO::cerr << "CHROMA: caught cast error" << std::endl;
      QDP_abort(1);
    } catch (std::bad_alloc) {
      std::cerr << "CHROMA: caught bad memory allocation" << std::endl;
      QDP_abort(1);
    } catch (const std::string& e) {
      QDPIO::cerr << "CHROMA: Caught Exception: " << e << std::endl;
      QDP_abort(1);
    } catch (std::exception& e) {
      QDPIO::cerr << "CHROMA: Caught standard library exception: " << e.what() << std::endl;
      QDP_abort(1);
    } catch (...) {
      std::cerr << "CHROMA: caught generic exception during gaugeInit" << std::endl;
      throw;
    }
    swatch.stop();
    QDPIO::cout << "CHROMA: Gauge field successfully read: time= "
                << swatch.getTimeInSeconds() << " secs" << std::endl;

    XMLBufferWriter config_xml;
    config_xml << gauge_xml;
    write(xml_out, "Config_info", gauge_xml);

    swatch.reset();
    swatch.start();
    MesPlq(xml_out, "Observables", u);
    swatch.stop();
    QDPIO::cout << "CHROMA: initial plaquette measurement time=" << swatch.getTimeInSeconds() << " secs" << std::endl;

    try {
      swatch.reset();
      swatch.start();
      std::istringstream Measurements_is(input.param.inline_measurement_xml);
      XMLReader MeasXML(Measurements_is);
      multi1d<Handle<AbsInlineMeasurement>> the_measurements;
      read(MeasXML, "/InlineMeasurements", the_measurements);

      QDPIO::cout << "CHROMA: There are " << the_measurements.size() << " measurements " << std::endl;

      InlineDefaultGaugeField::reset();
      InlineDefaultGaugeField::set(u, config_xml);

      push(xml_out, "InlineObservables");
      xml_out.flush();

      swatch.stop();
      QDPIO::cout << "CHROMA: parsing inline measurements time=" << swatch.getTimeInSeconds() << " secs" << std::endl;
      QDPIO::cout << "CHROMA: Doing " << the_measurements.size() << " measurements" << std::endl;
      swatch.reset();
      swatch.start();
      unsigned long cur_update = 0;
      swatch.stop();

      QDPIO::cout << "CHROMA: measurements: time= " << swatch.getTimeInSeconds() << " secs" << std::endl;
      pop(xml_out); // InlineObservables
      InlineDefaultGaugeField::reset();
    } catch (std::bad_cast) {
      QDPIO::cerr << "CHROMA: caught cast error" << std::endl;
      QDP_abort(1);
    } catch (std::bad_alloc) {
      std::cerr << "CHROMA: caught bad memory allocation" << std::endl;
      QDP_abort(1);
    } catch (const std::string& e) {
      QDPIO::cerr << "CHROMA: Caught Exception: " << e << std::endl;
      QDP_abort(1);
    } catch (const char* e) {
      QDPIO::cout << "CHROMA: Caught const char * exception: " << e << std::endl;
      QDP_abort(1);
    } catch (std::exception& e) {
      QDPIO::cerr << "CHROMA: Caught standard library exception: " << e.what() << std::endl;
      QDP_abort(1);
    } catch (...) {
      std::cerr << "CHROMA: caught generic exception during measurement" << std::endl;
      throw;
    }

    if (list_index == Chroma::getInputFileList().size() - 1) {
      pop(xml_out); // chroma
    }

    if (Chroma::getInputFileList().size() > 1) {
      TheNamedObjMap::Instance().erase_all();
    }
  }

  if (Chroma::getInputFileList().size() > 1) {
    QDPIO::cout << "CHROMA: total number of input files processed = " << Chroma::getInputFileList().size() << std::endl;
  }
  // from eloy:
  // ---------------------------------------------------  //
  // const auto& sdb = SB::openColorvecStorage(filename);
  // const auto& colorvec_rb = SB::getColorvecs<SB::Complex>(sdb, u,
  // decay_dir, t_slice, 1, num_vecs, "cxyzXnt");
  // const auto& colorvec = SB::detail::toNaturalOrdering(colorvec_rb,
  // t_slice).make_sure(none, SB::OnHost, SB::OnMaster);

  // Then, colorvec.data() is a cpu pointer to eigenvectors for the t_slice.
  // The ordering is color, x-dir, y-dir, z-dir, n eigenvector. So:

  // std::complex<double> *v = colorvec.data();
  // v[1] is the second element of color for the x,y,z=0,0,0 and eigenvector
  // zero.
  // ------------------------------------------------------------------ //

  // COLORVEC SDB -> TXT 
  // need to extract values of the eigenpairs 
  int from_tslice = 0;
  int n_colorvecs = 128;
  int n_tslices = 96;
  int decay_dir = 3;
  multi1d<LatticeColorMatrix> u(Nd);
  std::vector<std::string> colorvec_file = {sdb_file};
  std::cout << "Opening .sdb file: " << colorvec_file[0] << std::endl;
  
  const auto& sdb = SB::openColorvecStorage(colorvec_file);

  for (int t = from_tslice; t < n_tslices; ++t) {
    std::cout << "Processing time slice: " << t << std::endl;
    const auto& colorvec_rb = SB::getColorvecs<SB::Complex>(sdb, u, decay_dir, t, 1, num_vecs, "cxyzXnt");
    const auto& colorvec = SB::detail::toNaturalOrdering(colorvec_rb, t)
                              .make_sure(SB::none, SB::OnHost, SB::OnMaster);

    std::complex<double> *v = colorvec.data();
    // Print eigenvectors and eigenvalues
    for (int n = 0; n < n_colorvecs; ++n) {
      QDPIO::cout << "  Eigenvector " << n << ": (" << v[n].real() << ", " << v[n].imag()
                  << std::endl;
    }
  }

  snoop.stop();
  QDPIO::cout << "Time to read " << n_colorvecs << " colorvecs from " << n_tslices
              << " time slices: " << snoop.getTimeInSeconds() << " secs" << std::endl;
  QDPIO::cout << "CHROMA: total time = " << snoop.getTimeInSeconds() << " secs" << std::endl;
  QDPIO::cout << "CHROMA: ran successfully" << std::endl;

  END_CODE();
  Chroma::finalize();
  exit(0);
}
