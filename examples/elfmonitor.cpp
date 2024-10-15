#include "../include/elfplusplus.h"

#include "../include/platform_helper.h"
#include <unistd.h>
#include <sys/types.h>
#include <chrono>
#include <string>
#include <map>
#include <vector>
#include <set>
#include <memory>
#include <mutex>    
#include <thread>
#include <iostream>
#include <functional>
#include <algorithm>
#include <atomic>
#include <signal.h>
#include <sys/wait.h>
#include <fcntl.h>

static volatile bool sigint = false;
 
//main
int main(int argc, char **argv) {   
    std::cout << "ELFMonitor++" << std::endl;
    //open elf file and walk 
    //use elf.h structures, without the library first: 
    signal(SIGINT, &on_sigint);
	//siginterrupt is deprecated, use sigaction instead
    //siginterrupt(SIGINT, true);
    sigaction(SIGINT, NULL, NULL); 
    //collect installed packages
    fill_packages_from_system("elfpp.db");
    //netlink helper
    elfpp::netlink_helper nl_helper;
    //register handler
    auto handler = [](const std::string& path_to_check)->bool {
        //set pink
        std::cout << "\033[1;35m[+]new path: \033[0m" << path_to_check << std::endl; 
        
        return true;
    };
    nl_helper.register_handler(handler);
    
    std::cout << "[+]starting netlink listener" << std::endl;
    nl_helper.proc_listen(true);
    nl_helper.run();
    std::cout << "[+]done" << std::endl;
    return 0;
}





