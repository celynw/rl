#include <iostream>
#include <string>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
// #include <pybind11/complex.h>
// #include <pybind11/functional.h>
// #include <pybind11/chrono.h>

#include <esim/esim/event_simulator.hpp>
#include <esim/common/types.hpp>

//==================================================================================================
class SimulatorBridge {
	//----------------------------------------------------------------------------------------------
	protected:
		// std::string model;
		const double contrast_threshold_pos;
		const double contrast_threshold_neg;
		const double contrast_threshold_sigma_pos;
		const double contrast_threshold_sigma_neg;
		const int64_t refractory_period_ns;
		const bool use_log_image;
		const double log_eps;
		const bool simulate_color_events;
		event_camera_simulator::EventSimulator* simulator;

	//----------------------------------------------------------------------------------------------
	public:
		SimulatorBridge(
			const double &contrast_threshold_pos,
			const double &contrast_threshold_neg,
			const double &contrast_threshold_sigma_pos,
			const double &contrast_threshold_sigma_neg,
			const int64_t &refractory_period_ns,
			const bool &use_log_image,
			const double &log_eps,
			const bool &simulate_color_events
		);
		event_camera_simulator::Events img2events(pybind11::array_t<uint8_t>& np_img, event_camera_simulator::Time time);
};


//==================================================================================================
SimulatorBridge::SimulatorBridge(
	const double &contrast_threshold_pos,
	const double &contrast_threshold_neg,
	const double &contrast_threshold_sigma_pos,
	const double &contrast_threshold_sigma_neg,
	const int64_t &refractory_period_ns,
	const bool &use_log_image,
	const double &log_eps,
	const bool &simulate_color_events
) :
	contrast_threshold_pos(contrast_threshold_pos),
	contrast_threshold_neg(contrast_threshold_neg),
	contrast_threshold_sigma_pos(contrast_threshold_sigma_pos),
	contrast_threshold_sigma_neg(contrast_threshold_sigma_neg),
	refractory_period_ns(refractory_period_ns),
	use_log_image(use_log_image),
	log_eps(log_eps),
	simulate_color_events(simulate_color_events) {
	const event_camera_simulator::EventSimulator::Config config{
		contrast_threshold_pos, // Contrast threshold (positive): double contrast_threshold_pos
		contrast_threshold_neg, // Contrast threshold (negative): double contrast_threshold_neg
		contrast_threshold_sigma_pos, // Standard deviation of contrast threshold (positive): double contrast_threshold_sigma_pos = 0.021
		contrast_threshold_sigma_neg, // Standard deviation of contrast threshold (negative): double contrast_threshold_sigma_neg = 0.021
		refractory_period_ns, // Refractory period (time during which a pixel cannot fire events just after it fired one), in nanoseconds: int64_t refractory_period_ns
		use_log_image, // Whether to convert images to log images in the preprocessing step: const bool use_log_image
		log_eps, // Epsilon value used to convert images to log: L = log(eps + I / 255.0): const double log_eps
		simulate_color_events, // Whether to simulate color events or not (default: false): const bool simulate_color_events
		// const double exposure_time_ms = 10.0, // Exposure time in milliseconds, used to simulate motion blur
		// const bool anonymous = false, // Whether to set a random number after the /ros_publisher node name (default: false)
		// const int32_t random_seed = 0 // Random seed used to generate the trajectories. If set to 0 the current time(0) is taken as seed.
	};
	this->simulator = new event_camera_simulator::EventSimulator(config);
}


//==================================================================================================
event_camera_simulator::Events SimulatorBridge::img2events(pybind11::array_t<uint8_t>& np_img, event_camera_simulator::Time time) {
	// Convert numpy array to OpenCV image
	pybind11::buffer_info buffer = np_img.request();
	auto ptr = static_cast<double *>(buffer.ptr);
	cv::Mat cv_img(buffer.shape[0], buffer.shape[1], CV_8UC3, (unsigned char*)buffer.ptr);

	// Convert OpenCV image to what ESIM needs
	// FIX check if X and Y are right way around
	event_camera_simulator::ColorImagePtr esim_img_ptr(new event_camera_simulator::ColorImage(cv::Size(buffer.shape[0], buffer.shape[1])));
	cv_img.convertTo(*esim_img_ptr, cv::DataType<ImageFloatType>::type, 1. / 255.);

	// Call the simulator
	event_camera_simulator::Events events = this->simulator->imageCallback(*esim_img_ptr, time);

	// DEBUG
	// for (const event_camera_simulator::Event& event : events)
	// 	std::cout << "[" << event.x << ", " << event.y << "] " << event.t << " : " << event.pol << std::endl;

	return events;
}


//==================================================================================================
PYBIND11_MODULE(esimcpp, m) {
	m.doc() = "ESIM simulator bridge for python"; // optional module docstring

	pybind11::class_<SimulatorBridge>(m, "SimulatorBridge")
		.def(pybind11::init<const double &, const double &, const double &, const double &, const int64_t &, const bool &, const double &, const bool &>())
		.def("img2events", &SimulatorBridge::img2events);


	// Needed to be able to return Events type data
	pybind11::class_<event_camera_simulator::Event>(m, "Event")
		.def(pybind11::init<uint16_t, uint16_t, event_camera_simulator::Time, bool>())
		.def_readwrite("x", &event_camera_simulator::Event::x)
		.def_readwrite("y", &event_camera_simulator::Event::y)
		.def_readwrite("t", &event_camera_simulator::Event::t)
		.def_readwrite("pol", &event_camera_simulator::Event::pol)
		.def("__repr__", [](const event_camera_simulator::Event& e) {
			std::ostringstream repr;
			repr << "{";
			repr << "x: " << e.x << ", ";
			repr << "y: " << e.y << ", ";
			repr << "t: " << e.t << ", ";
			repr << "pol: " << e.pol;
			repr << "}";

			return repr.str();
		});
}


//==================================================================================================
int main() {
	std::cout << "Main function" << std::endl;

	// TODO generate numpy array as input
	// img2events(img);

	return EXIT_SUCCESS;
}
