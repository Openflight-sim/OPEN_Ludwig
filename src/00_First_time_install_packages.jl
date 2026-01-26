# FILE: _First_time_run.jl
import Pkg

function install_dependencies()
    println("==========================================================")
    println("      LBM SOLVER | DEPENDENCY INSTALLER                   ")
    println("==========================================================")

    # List of external packages required by the solver
    dependencies = [
        "KernelAbstractions", # For backend-agnostic kernel writing
        "CUDA",               # For GPU acceleration
        "Adapt",              # For data transfer between CPU/GPU
        "StaticArrays",       # For high-performance small arrays (SVector)
        "YAML",               # For reading configuration files
        "WriteVTK"            # For exporting results to ParaView
    ]

    println("[Installer] Activating project environment...")
    # Activates the current directory as the project environment
    # This creates a Project.toml and Manifest.toml if they don't exist
    #Pkg.activate(".") 

    println("[Installer] The following packages will be installed:")
    for dep in dependencies
        println("  - $dep")
    end
    println("----------------------------------------------------------")

    try
        # Install packages
        Pkg.add(dependencies)
        
        println("\n[Installer] Packages added successfully.")
        println("[Installer] Precompiling (this may take a few minutes)...")
        
        # Precompile to make the first run faster
        Pkg.precompile()
        
        println("\n[Success] Environment is ready!")
        println("----------------------------------------------------------")
        println("To run the solver:")
        println("1. Open a terminal in this folder")
        println("2. Run: julia --project=. src/main.jl")
        
    catch e
        println("\n[Error] Installation failed!")
        println(e)
    end
end

# Run the installer
install_dependencies()