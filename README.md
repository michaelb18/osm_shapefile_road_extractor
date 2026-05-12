<!-- Project Header -->
<br />
<div align="center" id="readme-top">
  <h3 align="center">🛣️ OSM Shapefile Road Extractor </h3>

  <p align="center">
    Extract OSM roads within a radius, filter by type, calculate distances, compare road networks
    across different dates, and export as Shapefile.
    <br />
    
  </p>
</div>


<!-- Table of Contents -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#license">License</a></li>

  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

A streamlit web application that runs locally and uses the OpenStreetMap (OSM) dataset to extract, analyse, and visualise road networks. 

Users can specify a geographic location and radius to fetch road data, filter by road type, calculate distances, and compare network changes across different dates. The application also provides functionality to export the processed road data as a Shapefile (`.shp`) for further spatial analysis.

### Features

#### Road extraction
Query road networks by coordinate and radius **(100m–5km)**. Fetches current OSM data by default, or any historical snapshot back to September 2012 using the Overpass `[date:]` filter.

Default setting is **Khartoum, Sudan**, as shown below: 
![Khartoum, Sudan](/images/khartoum_default.png)
- lat/lon: 15.552700, 32.532400
- radius: 1km
- date: Current day's OSM data
→ 303 road segments found (as of 12-05-2026)

Clicking a preset from the presets menu auto-fills the location and radius fields. Presets are saved locally in `presets.csv` (**NB**: .gitignore) and can be added to by saving a new preset.

#### Interactive map
Roads rendered on a CartoDB basemap, colour-coded by highway type (motorway, primary, residential, etc.) with tooltips showing road name and classification.

Types of roads (available for filtering):
  * motorway
  * trunk
  * primary
  * secondary
  * tertiary
  * residential
  * service
  * footway
  * cycleway
  * path

#### Nearest road finder
Input any coordinate to find the closest road segment. Displays distance in metres, road type, and road name, with a connector line drawn between the query point and the snap point on the road. Includes a DMS → decimal degrees converter.

#### Date comparison
Fetch the same area at two different dates and diff the results by OSM way ID. Highlights added roads (green), removed roads (red), and unchanged roads (grey) on a single map. Reports segment counts and total kilometres added, removed, and net. Both result sets exportable as Shapefiles.

#### Interval analysis
Sample the road network at regular intervals (monthly, quarterly, every 6 months, or yearly) across a date range. Produces a timeline of total network length, segment count, and additions/removals at each snapshot. Includes a road type composition area chart and full CSV export.

#### Filtering
Filter the fetched network by highway type and minimum segment length. Results shown in a live-updating attribute table.

#### Statistics
Total length, segment count, average and longest segment, and a length breakdown by road type with bar chart.

#### Export
Download results as a zipped ESRI Shapefile (all 5 components) or GeoJSON. Filtered subsets and comparison results (added/removed) each exportable independently.

#### Preset coordinates
Load saved coordinates from a local `presets.csv` file (gitignored). Selecting a preset auto-fills the lat/lon and date fields. All presets visualised as pins on an overview map.



### Built With

Built using the following libraries:

* [![Streamlit][Streamlit]][Streamlit-url]
* [![GeoPandas][GeoPandas]][GeoPandas-url]
* [![Folium][Folium]][Folium-url]
* [![Shapely][Shapely]][Shapely-url]
* [![Pandas][Pandas]][Pandas-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Setup
Copy `presets.csv.template` to `presets.csv` and fill in your coordinates.  
`presets.csv` is excluded from version control and should never be committed.



### Prerequisites

Install requirements using venv (macOS/Linux):

    python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

Install requirements using venv (Windows):

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

For new terminal run this first (macOS/Linux):

    source .venv/bin/activate

For new terminal run this first (Windows):

    .venv\Scripts\activate


### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/clarenceaurelius/osm_shapefile_road_extractor.git
   ```
   
2. Change git remote url to avoid accidental pushes to base project
   ```sh
   git remote set-url origin github_username/repo_name
   git remote -v # confirm the changes
   ```

3. Install requirements (as shown in Prerequisites section above)

4. **Optional**: Copy `presets.csv.template` to `presets.csv` and fill in your coordinates.  

5. Run the app
   ```sh
   streamlit run shap_extract.py
   ```


**NB**. `presets.csv` is excluded from version control and should never be committed.
<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- CONTRIBUTING -->
## Contributing

Any contributions are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Clarence Claus - [GitHub](https://github.com/clarenceaurelius) - [LinkedIn](https://linkedin.com/in/clarenceaureliusclaus)

Project Link: [https://github.com/clarenceaurelius/osm_shapefile_road_extractor](https://github.com/clarenceaurelius/osm_shapefile_road_extractor)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the GNU General Public License v3.0. See [`LICENSE`](LICENSE) for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->

[Streamlit]: https://img.shields.io/badge/Streamlit-35495E?style=for-the-badge&logo=streamlit&logoColor=61DAFB
[Streamlit-url]: https://streamlit.io/
[GeoPandas]: https://img.shields.io/badge/GeoPandas-20232A?style=for-the-badge&logo=geopandas&logoColor=61DAFB
[GeoPandas-url]: https://geopandas.org/
[Folium]: https://img.shields.io/badge/Folium-35495E?style=for-the-badge&logo=folium&logoColor=61DAFB
[Folium-url]: https://folium.readthedocs.io/
[Shapely]: https://img.shields.io/badge/Shapely-35495E?style=for-the-badge&logo=shapely&logoColor=61DAFB
[Shapely-url]: https://shapely.readthedocs.io/
[Pandas]: https://img.shields.io/badge/Pandas-35495E?style=for-the-badge&logo=pandas&logoColor=61DAFB
[Pandas-url]: https://pandas.pydata.org/
